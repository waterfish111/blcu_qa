# -*- coding: utf-8 -*-
"""从真实语料里生成评测题（模板化，但每条的金标都是原文摘出来的）。

为什么要脚本生成而不是手写：
    30 道题时 1 道题 = 4 个百分点，指标不可信；手写 50 道又容易写错数字。
    这里的事实全部从 chunks.jsonl 里正则抽出来，gold_span 保证是原文子串，
    mk_gold.py 一定能解析到对应块——不会出现"题写错了导致假性下降"。

用法：
    python scripts/gen_questions.py --dry-run     # 只看会生成什么
    python scripts/gen_questions.py --write       # 追加进 eval/questions.jsonl
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHUNKS = ROOT / "data" / "chunks.jsonl"
QUESTIONS = ROOT / "eval" / "questions.jsonl"

# 三个版本的写法不一样，分开抽
RE_CREDIT_2026 = re.compile(r"准予毕业总学分：\s*(\d+)\s*学分")
RE_CREDIT_OLD = re.compile(r"准予毕业课程学分：\s*(\d+)\s*学分")
RE_HOURS = re.compile(r"课程总学时：\s*(\d+)\s*学时")
RE_DEGREE = re.compile(r"授予([^\s。；，、]{2,8}?)学士学位")
RE_WEEKS = re.compile(r"教学总周数：\s*(\d+)\s*周")

# 模块学分：2026 是表格行（"公共必修课\n37 学分"），2025/2024 是一句话
RE_MODULE = re.compile(
    r"(公共必修课|公共选修课|外语必修课|通识教育选修课|通识教育核心课|通识教育基础课|"
    r"专业必修课|专业核心课|专业基础课|专业选修课)\s*(\d+)\s*学分"
)
# 有些页是"一个页面写好几个专业"的（页眉只挂了其中一个专业名），
# 抽出来的数字到底是哪个专业的会分不清 —— 这类页面直接跳过，不拿来出题。
RE_OTHER_MAJOR = re.compile(r"([\u4e00-\u9fa5（）()]{2,20}?专业)\s*(?:准予)?(?:毕业)?[^\d。；\n]{0,6}?(\d+)\s*(?:学分|学时)")

DOC_LABEL = {
    "blcu_plan_2026": "2026 级",
    "blcu_plan_2025": "2025 级",
    "blcu_plan_2024_vol1": "2024 级",
    "blcu_plan_2024_vol2": "2024 级",
}
COHORT = {
    "blcu_plan_2026": "2026级",
    "blcu_plan_2025": "2025级",
    "blcu_plan_2024_vol1": "2024级",
    "blcu_plan_2024_vol2": "2024级",
}
# 每个文档生成多少道、什么类型
QUOTA = {
    "blcu_plan_2026": [("总学分", 4), ("模块学分", 4), ("学位", 3)],
    "blcu_plan_2025": [("总学分", 3), ("总学时", 3), ("模块学分", 2), ("学位", 2)],
    "blcu_plan_2024_vol1": [("总学分", 3), ("总学时", 3), ("模块学分", 2)],
    "blcu_plan_2024_vol2": [("总学分", 2), ("总学时", 2), ("模块学分", 1), ("学位", 1)],
}


def major_of(section_path):
    return (section_path or "").split(">")[0].strip()


def load_chunks():
    out = []
    with open(CHUNKS, encoding="utf-8-sig") as f:
        for line in f:
            out.append(json.loads(line))
    return out


def is_multi_major(text, major):
    """页面里出现了"别的专业 + 数字"的写法，说明这页不止一个专业，无法判定数字归属。"""
    core = re.sub(r"[（(].*?[）)]", "", major)  # 去掉括号里的方向名再比
    for m in RE_OTHER_MAJOR.finditer(text):
        other = m.group(1)
        if core and core.rstrip("专业") and core.rstrip("专业") not in other:
            return True
    return False


def collect_facts(chunks):
    """按 (doc, 专业) 汇总可考的事实。只收"毕业要求/培养方案基本框架"这类块。"""
    facts = {}
    for c in chunks:
        path = c.get("section_path") or ""
        if "毕业要求" not in path and "基本框架" not in path:
            continue
        major = major_of(path)
        if not major or "未索引" in path:
            continue
        text = c["text"]
        if is_multi_major(text, major):
            continue
        slot = facts.setdefault((c["doc_id"], major), {"chunk_id": c["chunk_id"], "path": path})
        m = RE_CREDIT_2026.search(text) or RE_CREDIT_OLD.search(text)
        if m and "credits" not in slot:
            slot["credits"] = (m.group(1), m.group(0).replace("：", "：").strip())
        if "hours" not in slot:
            m = RE_HOURS.search(text)
            if m:
                slot["hours"] = (m.group(1), m.group(0).strip())
        if "degree" not in slot:
            m = RE_DEGREE.search(text)
            if m:
                slot["degree"] = (m.group(1), m.group(0).strip())
        if "weeks" not in slot:
            m = RE_WEEKS.search(text)
            if m:
                slot["weeks"] = (m.group(1), m.group(0).strip())
        for m in RE_MODULE.finditer(text):
            slot.setdefault("modules", {})[m.group(1)] = (m.group(2), m.group(0).replace("\n", "").strip())
    return facts


def existing_keys(rows):
    """已经问过的 (doc, 专业, 主题)，避免重复出题。"""
    keys = set()
    for r in rows:
        for doc in r.get("gold_doc_ids") or []:
            keys.add((doc, r.get("gold_path", "").strip(), r.get("type", "")))
            keys.add((doc, r.get("gold_path", "").strip(), "any"))
    return keys


def build_question(doc, major, topic, fact, qid):
    label = DOC_LABEL[doc]
    base = {
        "id": qid,
        "cohort": COHORT[doc],
        "expect_abstain": False,
        "gold_path": major,
        "gold_doc_ids": [doc],
    }
    if topic == "总学分":
        num, span = fact["credits"]
        base.update(
            q=f"{label}{major}准予毕业总学分是多少？",
            type="直接条款",
            gold_answer=f"{num} 学分",
            gold_span=span,
            note="同一段话在几十个专业里几乎一字不差，考的是'专业维度'能不能分对",
        )
    elif topic == "总学时":
        num, span = fact["hours"]
        base.update(
            q=f"{label}{major}的课程总学时是多少？",
            type="表格数值",
            gold_answer=f"{num} 学时",
            gold_span=span,
            note="学时数字只在毕业要求那一段出现，易与'教学总周数'混",
        )
    elif topic == "学位":
        deg, span = fact["degree"]
        base.update(
            q=f"{label}{major}毕业后授予什么学士学位？",
            type="直接条款",
            gold_answer=f"{deg}学士学位",
            gold_span=span,
            note="工科/文科专业授予的学位不同，考专业维度",
        )
    elif topic == "模块学分":
        name, (num, span) = fact
        base.update(
            q=f"{label}{major}的{name}要求多少学分？",
            type="表格数值",
            gold_answer=f"{num} 学分",
            gold_span=span,
            note="学分构成里的单项，同一专业内有多个相近数字，考细粒度",
        )
    else:
        raise ValueError(topic)
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    chunks = load_chunks()
    facts = collect_facts(chunks)
    rows = [json.loads(l) for l in open(QUESTIONS, encoding="utf-8-sig")]
    used = existing_keys(rows)
    used_majors = {(d, p) for (d, p, t) in used if t == "any"}

    new = []
    nid = max(int(r["id"][1:]) for r in rows) + 1
    for doc, quotas in QUOTA.items():
        pool = {m: f for (d, m), f in facts.items() if d == doc}
        for topic, quota in quotas:
            picked = 0
            for major in sorted(pool):
                if picked >= quota:
                    break
                if major in {m for (d, m) in used_majors if d == doc}:
                    continue
                fact = pool[major]
                if topic == "总学分" and "credits" not in fact:
                    continue
                if topic == "总学时" and "hours" not in fact:
                    continue
                if topic == "学位" and "degree" not in fact:
                    continue
                if topic == "模块学分":
                    mods = [k for k in ("专业核心课", "外语必修课", "公共必修课", "专业选修课") if k in fact.get("modules", {})]
                    if not mods:
                        continue
                    name = mods[picked % len(mods)]
                    q = build_question(doc, major, topic, (name, fact["modules"][name]), f"q{nid:03d}")
                else:
                    q = build_question(doc, major, topic, fact, f"q{nid:03d}")
                new.append(q)
                nid += 1
                picked += 1
                used_majors.add((doc, major))

    print(f"将新增 {len(new)} 道题（现有 {len(rows)} 道，合计 {len(rows) + len(new)}）")
    for q in new:
        print(f"  {q['id']} [{q['type']}] {q['q']}  ->  {q['gold_answer']}")

    if args.write:
        with open(QUESTIONS, "a", encoding="utf-8") as f:
            for q in new:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")
        print(f"已追加到 {QUESTIONS}")
        print("别忘了：python scripts/mk_gold.py --write --max-ids 8")


if __name__ == "__main__":
    main()
