"""把评测集里的 gold_span 解析成 gold_chunk_ids。

为什么要有这一步：手写 30 道题的正确答案容易，但手写"答案在哪个块"几乎不可能
（块号是切块程序生成的，改一次切块策略就全变）。所以评测集里只记一段"答案原文"
（gold_span），由本脚本去 data/chunks.jsonl 里找出包含这段原文的块。

用法：
    python scripts/mk_gold.py                # 只报告，不写文件
    python scripts/mk_gold.py --write        # 回填 gold_chunk_ids 到 eval/questions.jsonl
    python scripts/mk_gold.py --write --show # 顺便打印每个块的开头，便于人工核对
"""

import argparse
import json
import os
import re
import sys
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUESTIONS = os.path.join(ROOT, "eval", "questions.jsonl")
CHUNKS = os.path.join(ROOT, "data", "chunks.jsonl")

FIELDS = [
    "id", "q", "type", "cohort", "expect_abstain", "gold_answer",
    "gold_span", "gold_path", "gold_doc_ids", "gold_chunk_ids", "note",
]


def norm(s):
    """归一化：全角转半角、去掉所有空白、去掉表格竖线。

    这样「思想道德与法治 | 48 | 0 | 48 | 3」和按单元格换行的写法
    「思想道德与法治\\n48\\n0\\n48\\n3」会归一化成同一个串，两种版式都能命中。
    """
    s = unicodedata.normalize("NFKC", s or "")
    s = re.sub(r"[\s|｜]+", "", s)
    return s


def load_jsonl(path):
    with open(path, encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=QUESTIONS)
    ap.add_argument("--chunks", default=CHUNKS)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--max-ids", type=int, default=5,
                    help="一道题最多记几个金标块（答案被切散时会命中多个）")
    args = ap.parse_args()

    if not os.path.exists(args.chunks):
        sys.exit("找不到 %s，先跑 scripts/build_index.py" % args.chunks)

    questions = load_jsonl(args.questions)
    chunks = load_jsonl(args.chunks)
    for c in chunks:
        c["_norm"] = norm(c.get("text", ""))

    unmatched, total_ids, report = [], 0, []
    for q in questions:
        if q.get("expect_abstain"):
            q["gold_chunk_ids"] = []
            report.append((q["id"], q["type"], 0, "应拒答，跳过"))
            continue
        span = norm(q.get("gold_span", ""))
        if len(span) < 6:
            unmatched.append((q["id"], "gold_span 太短或为空"))
            continue
        docs = set(q.get("gold_doc_ids") or [])
        cand = [c for c in chunks
                if (not docs or c.get("doc_id") in docs) and span in c["_norm"]]
        path_hint = (q.get("gold_path") or "").strip()
        narrowed = [c for c in cand if path_hint and path_hint in (c.get("section_path") or "")]
        if narrowed:
            cand, note = narrowed, "按 gold_path 过滤"
        elif path_hint and cand:
            note = "gold_path 未命中，退回未过滤结果"
        else:
            note = ""

        ids = [c["chunk_id"] for c in cand[:args.max_ids]]
        q["gold_chunk_ids"] = ids
        total_ids += len(ids)
        report.append((q["id"], q["type"], len(ids), note))
        if not ids:
            unmatched.append((q["id"], "没有块包含 gold_span：%s" % q.get("gold_span", "")[:40]))
        if args.show:
            for c in cand[:args.max_ids]:
                print("   %s | %s | %s" % (c["chunk_id"], c.get("section_path", ""),
                                           c["text"][:70].replace("\n", " ")))

    print("%-6s %-8s %5s  %s" % ("id", "题型", "块数", "说明"))
    print("-" * 78)
    for qid, qtype, n, note in report:
        print("%-6s %-8s %5d  %s" % (qid, qtype, n, note))

    by_type = {}
    for qid, qtype, n, _ in report:
        a = by_type.setdefault(qtype, [0, 0])
        a[0] += 1
        a[1] += 1 if n else 0
    print("-" * 78)
    for qtype, (cnt, ok) in by_type.items():
        print("%-8s 题数 %2d / 有金标 %2d" % (qtype, cnt, ok))
    print("共 %d 题，金标块合计 %d 个" % (len(questions), total_ids))

    if unmatched:
        print("\n未解析 %d 条：" % len(unmatched))
        for qid, why in unmatched:
            print("  %s %s" % (qid, why))

    if args.write:
        with open(args.questions, "w", encoding="utf-8", newline="\n") as f:
            for q in questions:
                out = {k: q.get(k) for k in FIELDS if k in q}
                for k, v in q.items():
                    out.setdefault(k, v)
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
        print("\n已回填 gold_chunk_ids -> %s" % args.questions)


if __name__ == "__main__":
    main()
