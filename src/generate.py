"""生成层：强约束提示词 + 引用校验 + 拒答。

核心约束（这三条是这个项目区别于"套壳聊天"的地方）：
  1. 只能使用给定片段作答，片段里没有就拒答
  2. 每个结论句后面必须带 [n]，n 对应片段编号
  3. 生成后校验每个 [n] 是否有句子支撑，校验不过重试一次，再不过就拒答
"""

import difflib
import json
import os
import re

ABSTAIN_TEXT = "文档中未涉及该问题，建议咨询教务处或查阅最新版学生手册。"

# 解析层留下的"这一页的表是矢量图形、没进文本层"的说明行
UNINDEXED_MARK = "【未索引】"

PROMPT = """你是北京语言大学教务规定的问答助手。只依据下面提供的资料片段回答，不得使用外部知识。

规则：
1. 资料里找不到答案时，直接回复「{abstain}」，不要猜测。
2. 每个结论句后必须标注来源编号，如 [1][2]，编号必须来自下方片段。
3. 涉及年级差异时，先说明适用哪个年级的方案，再给结论。
4. 答案控制在 200 字内，先给结论，再给依据。
5. 若片段里出现「{unindexed}」，说明那一页的表格是矢量图形、文字没进文本层：
   要如实说"这页的表格读不到"，并给出页码，不能凭印象补数字。

资料片段：
{context}

用户问题：{question}

回答："""

CITE_RE = re.compile(r"\[(\d+)\]")


def load_dotenv(path=None):
    """读 .env（一行 KEY=VALUE），已存在的环境变量不覆盖。

    不装 python-dotenv 也能用；.env 已在 .gitignore 里，key 不会进仓库。
    测试时可用环境变量 BLCU_DOTENV 指定别的文件。
    """
    path = path or os.environ.get("BLCU_DOTENV") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if not os.path.exists(path):
        return {}
    loaded = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            loaded[k] = v
            os.environ.setdefault(k, v)
    return loaded


def build_context(chunks):
    lines = []
    for i, c in enumerate(chunks, 1):
        head = "[%d] 《%s》%s %s" % (
            i, c.get("title", ""), c.get("section_path", ""), c.get("article_no", ""))
        lines.append(head.strip() + "\n" + c["text"])
    return "\n\n".join(lines)


def compose(chunks, question, model="deepseek-chat", temperature=0.1):
    prompt = PROMPT.format(abstain=ABSTAIN_TEXT, unindexed=UNINDEXED_MARK,
                           context=build_context(chunks), question=question)
    try:
        from openai import OpenAI
        load_dotenv()
        client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],
                        base_url="https://api.deepseek.com")
        resp = client.chat.completions.create(
            model=model, temperature=temperature,
            messages=[{"role": "user", "content": prompt}])
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return "__ERROR__ " + str(e)


def cited_indexes(answer):
    return sorted({int(x) for x in CITE_RE.findall(answer)})


NUM_UNIT = r"[一二两三四五六七八九十百千零]+[年月日个次门课分天周条人]"
DIGIT_RE = re.compile(r"\d+(?:\.\d+)?")

# 说明"查不到"的话不是主张，不该按"编内容"处理
HEDGE_RE = re.compile(
    r"(资料中未|文档中未|未涉及|未汇总|未给出|未列明|无法给出|无法确定|无法提供"
    r"|建议(?:咨询|查阅|向|结合)|另需注意|需要注意的是|存在不同表述|两处不一致|建议核对)")


def is_table_chunk(chunk):
    """表格行切出来的块：正文几乎全是数字和短词，逐字比对没有意义。

    判据很土但稳：带竖线（表格行用 | 分隔）或者"数字占比超过 18%"。
    """
    text = chunk.get("text") or ""
    if "|" in text or "｜" in text:
        return True
    if len(text) < 20:
        return False
    digits = sum(1 for ch in text if ch.isdigit())
    return digits / len(text) > 0.18


def source_text(chunk):
    """一块"出处"的全文：书名＋路径＋条款号＋正文。

    校验必须对着这份文本判，而不是只对着正文——因为提示词里给模型看的就是这些，
    模型答"2026级…"的依据往往在 section_path（"2026版培养方案"）里，只比正文会把
    正确答案判成编造。
    """
    return " ".join(str(chunk.get(k) or "") for k in ("title", "section_path", "article_no")) \
        + "\n" + (chunk.get("text") or "")


def check_citations(answer, chunks, question=""):
    """引用一致性校验（启发式，按句判，两条规则）：

    1. **数字对得上**：句中的阿拉伯数字和中文量词（"三 年"/"两 年"）必须在片段原文里出现。
       模型最常犯的错不是编内容，而是把"两年"写成"三年"、把 142 写成 147，字面几乎一样。
    2. **新增内容找得到**：把句子里"照抄提问"的部分（≥3 字的公共片段）剪掉，
       剩下的中文长词（≥3 字）必须能在片段原文里逐字找到——找不到就是凭空补的。

    为什么不用"二字组重合率 ≥45%"这种一看就简单的方法：实测会把正确答案大面积误杀。
    "金融学（数智金融）专业准予毕业总学分为149学分"这句话里，主语抄自提问、正文里没有，
    重合率只有 0.33–0.39，全被卡掉了。改成"剪掉提问后逐字找"，才既严又准。

    按"这句引用的片段合集"判：一句结论常常同时用两张卡（"学制四年，授予经济学学士学位[1][3]"）。

    返回对不上的引用编号列表（升序去重），空列表表示全部通过。
    """
    if not answer or answer.startswith("__ERROR__") or answer == ABSTAIN_TEXT:
        return []
    bad = set()
    sentences = [s for s in re.split(r"[。！？\n]", answer) if CITE_RE.search(s)]
    for s in sentences:
        idxs = [int(x) for x in CITE_RE.findall(s)]
        valid = [i for i in idxs if 1 <= i <= len(chunks)]
        bad.update(i for i in idxs if i not in valid)
        if not valid:
            continue
        source = "\n".join(source_text(chunks[i - 1]) for i in valid)
        table_only = all(is_table_chunk(chunks[i - 1]) for i in valid)
        # 表格行只有数字和短词，逐字比对一定误杀；这种句子只靠"数字对得上"把关
        if not _numbers_ok(s, source, question):
            bad.update(valid)
        elif not table_only and _unsupported(s, source, question):
            bad.update(valid)
    return sorted(bad)


def _flat(text):
    """去掉引用编号、标点、空白，只留汉字和字母数字，方便逐字比对。"""
    return re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9]", "", CITE_RE.sub("", text))


def _novel_text(sentence, question):
    """剪掉句子里照抄提问的部分，只留模型自己加的话。"""
    s, q = _flat(sentence), _flat(question)
    if not q:
        return s
    keep, last = [], 0
    for blk in difflib.SequenceMatcher(None, s, q, autojunk=False).get_matching_blocks():
        if blk.size >= 3:  # 3 字以上重合才算"照抄"，否则"学分"这种通用词会被误剪
            keep.append(s[last:blk.a])
            last = blk.a + blk.size
    keep.append(s[last:])
    return "".join(keep)


def _unsupported(sentence, source, question=""):
    """这句话是不是"凭空补的"：剪掉照抄提问的部分后，剩下的字有多少能在片段里找到。

    只看模型新加的内容，且用二字组覆盖率（>50% 算有据），不要求整句逐字出现——
    原文一句会被拆到两句答案里，整句比对会把正常回答判死。
    """
    novel = HEDGE_RE.sub("", _novel_text(sentence, question))
    if len(novel) < 3:
        return False  # 只多了两三个字（或全是"查不到"的说明），谈不上新增主张
    grams = {novel[i:i + 2] for i in range(len(novel) - 1)}
    src = _flat(source)
    hit = sum(1 for g in grams if g in src)
    return hit / len(grams) < 0.5


def _numbers_ok(sentence, source, question=""):
    """句中的数字必须在出处原文出现。

    中文数字要连量词一起比（"三年" vs "两年"）——只比单字"三"会被"第三章"这种误放行。
    """
    sentence = CITE_RE.sub("", sentence)  # 去掉 [1] 这类引用编号，否则数字会被误判
    from_question = set(re.findall(NUM_UNIT, question)) | set(DIGIT_RE.findall(question))
    must = (set(re.findall(NUM_UNIT, sentence)) | set(DIGIT_RE.findall(sentence))) - from_question
    return all(num in source for num in must)


def trim_unsupported(text, chunks, question=""):
    """把校验不过的句子整句剪掉，只留能对上原文的部分；一句都不剩就返回空串。

    为什么要剪而不是一律拒答：模型常见的失败是"主体答对了、尾巴补了一句没出处的"，
    整篇拒答对用户是净损失。剪完还剩内容，说明主结论有据。
    """
    kept = []
    for part in re.split(r"(?<=[。！？\n])", text):
        if not part.strip() or not CITE_RE.search(part):
            continue
        if check_citations(part, chunks, question=question):
            continue
        kept.append(part.strip())
    return "".join(kept)


UNINDEXED_RE = re.compile(r"第\s*(\d+)\s*页的表格（([^）]*)）")


def unindexed_hits(chunks):
    """挑出"表格没进文本层"的说明块，返回 [(chunk, 页码, 表名), ...]。"""
    out = []
    for c in chunks:
        text = c.get("text") or ""
        if UNINDEXED_MARK not in text:
            continue
        m = UNINDEXED_RE.search(text)
        if m:
            out.append((c, m.group(1), m.group(2)))
    return out


def unindexed_answer(hit):
    _, page, names = hit
    return ("这个问题要查的表格在 PDF 第 %s 页（%s），但这份文件把那张表画成了矢量图形，"
            "文字没有进入文本层，我读不到表格里的数字。可以翻 PDF 第 %s 页核对；"
            "如果只是想了解大致口径，可以改问 2026 或 2024 版培养方案里的同类表。"
            % (page, names, page))


def answer(question, retrieved, top_score, threshold=0.35, chunks_limit=5, debug=False):
    """检索片段 → 答案。reason 说明结论是从哪来的，方便评测时分开统计：

      no_evidence      重排分不够，直接拒答（没调模型，不花钱）
      unindexed        这条正好落在"表格是矢量图形、文本层没有"的那几页上
      citation_failed  答了两次，引用都对不上原文，改成拒答
      no_citation      答了但一句 [n] 都没标（等于来源不明）
      ok               正常，引用校验通过
    """
    chunks = retrieved[:chunks_limit]
    if not chunks or top_score < threshold:
        hits = unindexed_hits(retrieved[:10])
        if hits:
            best = max(hits, key=lambda h: h[0].get("score") or 0)
            score = best[0].get("score") or 0
            # 太弱的相关不算：否则任何"查不到"的题都会被推给某一页表
            if score >= 0.2 and score >= 0.6 * max(top_score, 1e-6):
                return {"answer": unindexed_answer(best), "abstained": True,
                        "reason": "unindexed", "used_chunk_ids": [], "citation_ok": True}
        return {"answer": ABSTAIN_TEXT, "abstained": True, "reason": "no_evidence",
                "used_chunk_ids": [], "citation_ok": True}

    text = compose(chunks, question)
    bad = check_citations(text, chunks, question=question)

    trimmed = False
    if bad:  # 引用对不上：重试一次；还不行就剪掉没出处的句子，剪光才拒答
        draft1 = text
        text = compose(chunks, question)
        bad = check_citations(text, chunks, question=question)
        if bad:
            cut = trim_unsupported(text, chunks, question)
            if cut and not check_citations(cut, chunks, question=question):
                text, trimmed = cut, True
            else:
                return {"answer": ABSTAIN_TEXT, "abstained": True,
                        "reason": "citation_failed",
                        "used_chunk_ids": [], "citation_ok": False,
                        "debug": {"bad_citations": bad, "draft": text,
                                  "draft_first": draft1} if debug else None}

    if text.startswith("__ERROR__"):
        return {"answer": text, "abstained": True, "reason": "api_error",
                "used_chunk_ids": [], "citation_ok": False}

    if ABSTAIN_TEXT in text:  # 模型自己判断"资料里没有"，这是正常拒答，不是没标引用
        # 如果拒答的理由是"这张表是矢量图形、读不到"，那就不是泛泛的"未涉及"，
        # 要说清楚，并把那句通用兜底话去掉，免得看上去像没查。
        unindexed = any(UNINDEXED_MARK in (c.get("text") or "")
                        for c in retrieved[:chunks_limit])
        if unindexed:
            text = text.replace(ABSTAIN_TEXT, "").strip() or text
            return {"answer": text, "abstained": True, "reason": "unindexed",
                    "used_chunk_ids": [], "citation_ok": True}
        return {"answer": text, "abstained": True, "reason": "model_abstained",
                "used_chunk_ids": [], "citation_ok": True}

    if not cited_indexes(text):
        return {"answer": text, "abstained": False, "reason": "no_citation",
                "used_chunk_ids": [], "citation_ok": False}

    used = [chunks[i - 1]["chunk_id"] for i in cited_indexes(text)
            if 1 <= i <= len(chunks)]
    return {"answer": text, "abstained": False, "reason": "ok",
            "used_chunk_ids": used, "citation_ok": True, "trimmed": trimmed}


if __name__ == "__main__":
    # 冒烟测试：不调 API，只验证提示词拼装与引用校验
    demo = [{"chunk_id": "d#c1", "title": "学生手册", "section_path": "第三章 学籍",
             "article_no": "第十二条", "text": "学生休学一般以一年为期，累计不得超过两年。"}]
    print(build_context(demo))
    print(check_citations("休学累计不得超过两年[1]。", demo))
    print(check_citations("休学累计不得超过三年[1]。", demo))
