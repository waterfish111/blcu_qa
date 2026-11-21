"""切块：三种结构感知策略 + 固定窗口基线，模式可切换，便于做消融对比。

  article  按「章-节-条」切（规章类文档）
  major    按「专业-小节」切（培养方案类文档，本项目的针对性改法）
  window   固定窗口滑窗（基线，对照用）
  auto     文书里有条款号走 article，没有则走 major

用法：
  python src/chunk.py --docs data/docs.jsonl --out data/chunks.jsonl --mode auto
"""

import argparse
import json
import re

ARTICLE_RE = re.compile(r"^\s*第\s*([一二三四五六七八九十百零〇\d]+)\s*条")
CHAPTER_RE = re.compile(r"^\s*第\s*([一二三四五六七八九十百零〇\d]+)\s*章\s*(.*)$")
SECTION_RE = re.compile(r"^\s*第\s*([一二三四五六七八九十百零〇\d]+)\s*节\s*(.*)$")

# 「计算机科学与技术专业」「人工智能专业（080717T）」「翻译专业（英语方向）」
MAJOR_RE = re.compile(
    r"^\s*([\u4e00-\u9fa5A-Za-z·\s（）()]{2,30}?专业)\s*"
    r"(?:[（(]\s*(?:专业代码[:：]?\s*)?([0-9A-Z]{3,8})\s*[)）])?\s*$"
)
# 专业标题里出现这些词的，多半不是专业名，排除掉
MAJOR_BLACKLIST = ("转专业", "辅修", "双学位", "专业方向", "专业设置", "专业介绍",
                   "各专业", "本专业", "该专业", "专业代码", "跨专业")

# 「一、培养目标」「（二）培养要求」「七、课程设置与学分要求」
SECTION_NO_RE = re.compile(
    r"^\s*(?:[一二三四五六七八九十]{1,3}\s*[、.．]\s*|（[一二三四五六七八九十]{1,3}）\s*)"
    r"([\u4e00-\u9fa5A-Za-z、与及/]{2,20})\s*$"
)

# 目录行：名称……12
TOC_RE = re.compile(r"[.．…]{3,}\s*\d+\s*$")
# 培养方案的页眉：有的只写书名，有的把当前专业一起带上
#   「北京语言大学 2026 版本科人才培养方案 | 数字经济专业」
#   「北京语言大学2024 版本科人才培养方案（上册）| 播音与主持艺术专业（中英双语）」
#   「北京语言大学本科人才培养方案（2025版）」
HEADER_RE = re.compile(
    r"^\s*北京语言大学.{0,30}?培养方案(?:[（(][^）)]{1,8}[）)])?"
    r"\s*(?:[|｜]\s*(?P<major>.+?))?\s*$"
)
MAJOR_MARK = "【专业】"
TABLE_MARK = "【表格】"
# 解析层发现"这一页有表、但表是矢量图形画的"时留下的说明行，单独成块
UNINDEXED_MARK = "【未索引】"
UNINDEXED_LABEL = "未索引表格"
# 每份培养方案开头的标题行：「经济学（数智分析）专业培养方案」「金融学专业第二学士学位培养方案」
TITLE_RE = re.compile(
    r"^\s*(?P<name>[^，,。：:]{2,30}?(?:专业|实验班|方向)(?:第二学士学位)?)\s*培养方案\s*$"
)


def clean_lines(text, emit_major=False):
    """去目录点导线与页眉页码，并把「专业边界」整理成一行标记。

    页眉每页都重复，还会把页码单独顶成一行，混进正文会污染切块；
    但页眉里的专业名恰好是「这块内容属于哪个专业」的可靠信号，
    所以 emit_major=True 时把它转成一行标记，只在专业切换时吐一次。

    2025/2024 版没有逐页专业名，靠每份培养方案开头的标题行
    「经济学（数智分析）专业培养方案」来定位专业边界。
    """
    out = []
    last_major = None
    pending_header = False

    def emit(name):
        nonlocal last_major
        name = name.strip()
        if emit_major and name and name != last_major:
            out.append(MAJOR_MARK + name)
            last_major = name

    lines = text.splitlines()
    for i, line in enumerate(lines):
        s = line.strip()
        m = HEADER_RE.match(s)
        if m:
            pending_header = True
            emit(m.group("major") or "")
            continue
        if pending_header:
            pending_header = False
            if re.fullmatch(r"\d{1,4}", s):
                continue
        m_title = TITLE_RE.match(s)
        if m_title:
            # 目录里也有同名标题，用「后面紧跟着专业名称/所属单位」区分正文标题
            nxt = [x.strip() for x in lines[i + 1:i + 5] if x.strip()]
            looks_like_body = any("专业名称" in x or "所属单位" in x for x in nxt)
            if looks_like_body:
                emit(m_title.group("name"))
            continue
        if TOC_RE.search(line):
            continue
        if not s:
            out.append("")
            continue
        out.append(line.rstrip())
    return out


def _window(body, max_chars, overlap):
    if len(body) <= max_chars:
        return [body]
    step = max(1, max_chars - overlap)
    return [body[i:i + max_chars].strip() for i in range(0, len(body), step) if body[i:i + max_chars].strip()]


def _pack_rows(body, max_chars):
    """按整行打包，保证「一行 = 一门课/一条要求」不会被切断。"""
    out, cur = [], []
    size = 0
    for line in body.splitlines():
        n = len(line) + 1
        if cur and size + n > max_chars:
            out.append("\n".join(cur))
            cur, size = [line], n
        else:
            cur.append(line)
            size += n
    if cur:
        out.append("\n".join(cur))
    return [c for c in out if c.strip()]


def split_window(text, max_chars, overlap, min_chars=0):
    """固定窗口基线：不看任何结构。"""
    out = []
    for piece in _window(re.sub(r"\s+", " ", text).strip(), max_chars, overlap):
        if len(piece) >= min_chars:
            out.append(("", "", piece))
    return out


def split_structured(text, max_chars, overlap, min_chars):
    """按「章-节-条」切。返回 [(article_no, section_path, content), ...]"""
    blocks = []
    chapter = section = ""
    buf, art = [], ""

    def flush():
        nonlocal buf, art
        body = "\n".join(buf).strip()
        if body:
            blocks.append((art, " > ".join(x for x in (chapter, section) if x), body))
        buf, art = [], ""

    for line in clean_lines(text):
        m_ch = CHAPTER_RE.match(line)
        m_se = SECTION_RE.match(line)
        m_ar = ARTICLE_RE.match(line)
        if m_ch:
            flush()
            chapter, section = ("第%s章 %s" % (m_ch.group(1), m_ch.group(2).strip())).strip(), ""
        elif m_se:
            flush()
            section = ("第%s节 %s" % (m_se.group(1), m_se.group(2).strip())).strip()
        elif m_ar:
            flush()
            art = "第%s条" % m_ar.group(1)
            buf = [line]
        else:
            buf.append(line)
    flush()

    out = []
    for art_no, path, body in blocks:
        for piece in _window(body, max_chars, overlap):
            out.append((art_no, path, piece))
    return out


def split_major(text, max_chars, overlap):
    """按「专业 > 小节」切：培养方案的正文都是这个骨架，条款号基本没有。

    表格行不切断：按行打包，避免把「学分 | 学时 | 学期」这样的行劈成两半。
    """
    blocks = []
    major = section = ""
    table_label = ""
    buf = []

    def flush():
        nonlocal buf, table_label
        body = "\n".join(buf).strip()
        if body:
            # 表格只是这一小节的附属物，路径里两段都留，方便回答「哪门课多少学分」
            label = " > ".join(x for x in (section, table_label) if x)
            blocks.append((" > ".join(x for x in (major, label) if x), body))
        buf = []
        table_label = ""

    lines = clean_lines(text, emit_major=True)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(MAJOR_MARK):
            # 页眉里的专业名 = 专业边界；一个专业里还嵌着《辅修专业管理办法》这类规章，
            # 它们用自己的「第X章」，所以要保留章级路径。
            flush()
            major = stripped[len(MAJOR_MARK):].strip()
            section, table_label = "", ""
            continue
        if stripped == TABLE_MARK:
            flush()
            table_label = "课程计划表"
            continue
        if stripped.startswith(UNINDEXED_MARK):
            # 这条说明要和正文分开：它自己就是一条"我没读到"的证据，
            # 混进课程段里会让人以为那张表已经索引过了。
            flush()
            blocks.append((" > ".join(x for x in (major, UNINDEXED_LABEL) if x), stripped))
            continue
        m_ch = CHAPTER_RE.match(stripped)
        if m_ch:
            flush()
            section = ("第%s章 %s" % (m_ch.group(1), m_ch.group(2).strip())).strip()
            table_label = ""
            continue
        m_major = MAJOR_RE.match(stripped)
        m_sec = SECTION_NO_RE.match(stripped)
        nxt = next((x.strip() for x in lines[i + 1:i + 3] if x.strip()), "")
        # 光凭「以专业结尾」会误伤，比如课程名「从技术走向科学的采矿工程专业」；
        # 真正的小节标题后面紧跟着「一、专业名称」，带专业代码时可以直接认。
        heading_like = bool(m_major) and (m_major.group(2) or "专业名称" in nxt
                                         or "所属单位" in nxt or bool(SECTION_NO_RE.match(nxt)))
        if heading_like and not any(w in stripped for w in MAJOR_BLACKLIST) and len(stripped) <= 30:
            flush()
            code = m_major.group(2)
            major = m_major.group(1).strip() + ("（%s）" % code if code else "")
            section, table_label = "", ""
        elif m_sec:
            flush()
            section = stripped
        else:
            buf.append(line)
    flush()

    out = []
    for path, body in blocks:
        pieces = _pack_rows(body, max_chars) if "|" in body else _window(body, max_chars, overlap)
        for piece in pieces:
            out.append(("", path, piece))
    return out


def _merge_short(parts, min_chars):
    """无条款号的碎片并入上一块；有条款号的短条款保持独立，保证引用能精确到条。

    只在同一小节内合并：跨小节的碎块合并会把「专业名 > 小节」定位弄丢，
    检索出来也没法说明是哪一段。
    """
    merged = []
    for art_no, path, body in parts:
        same_section = bool(merged) and merged[-1][1] == path
        if merged and same_section and not art_no and len(body) < min_chars:
            merged[-1][2] = merged[-1][2] + "\n" + body
        else:
            merged.append([art_no, path, body])
    return merged


def chunk_doc(doc, max_chars=400, overlap=80, min_chars=40, mode="auto"):
    doc_id = doc["doc_id"]
    text = doc.get("text", "")

    if mode == "auto":
        # 只看"有没有出现过第X条"会把培养方案误判成规章：一本培养方案里往往
        # 内嵌一份《…管理办法》，整本书就被按章切碎、其余全成无路径的窗口块。
        doc_type = (doc.get("doc_type") or "").strip()
        if doc_type in ("培养方案", "plan", "training_plan"):
            mode = "major"
        else:
            lines = [l for l in clean_lines(text) if l.strip()]
            hits = sum(1 for l in lines if ARTICLE_RE.match(l))
            ratio = hits / float(max(1, len(lines)))
            mode = "article" if (ratio >= 0.05 and hits >= 3) or ratio >= 0.3 else "major"

    if mode == "article":
        parts = _merge_short(split_structured(text, max_chars, overlap, min_chars), min_chars)
    elif mode == "major":
        parts = _merge_short(split_major(text, max_chars, overlap), min_chars)
    elif mode == "window":
        parts = split_window(text, max_chars, overlap, min_chars)
    else:
        raise ValueError("unknown mode: %s" % mode)

    chunks = []
    for i, (art_no, path, body) in enumerate(parts):
        chunks.append({
            "chunk_id": "%s#c%03d" % (doc_id, i),
            "doc_id": doc_id,
            "title": doc.get("title", ""),
            "cohort": doc.get("cohort", ""),
            "status": doc.get("status", "active"),
            "article_no": art_no,
            "section_path": path,
            "text": body,
        })
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="data/docs.jsonl")
    ap.add_argument("--out", default="data/chunks.jsonl")
    ap.add_argument("--max-chars", type=int, default=400)
    ap.add_argument("--overlap", type=int, default=80)
    ap.add_argument("--mode", default="auto", choices=["auto", "article", "major", "window"])
    args = ap.parse_args()

    total = 0
    with open(args.docs, "r", encoding="utf-8-sig") as fin, \
            open(args.out, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            for c in chunk_doc(doc, args.max_chars, args.overlap, mode=args.mode):
                fout.write(json.dumps(c, ensure_ascii=False) + "\n")
                total += 1
    print("chunks written:", total, "->", args.out)


if __name__ == "__main__":
    main()
