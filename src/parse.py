"""解析层：PDF -> 文本 + 表格，网页 -> 正文。

策略：
  1. 先抽文本；若单页平均字符数过低，判定为扫描件，直接报错让人换文件（不硬做 OCR）
  2. 表格单独抽取，转成「表头 | 行 | 行」的线性文本，作为独立 chunk 入库
  3. 网页用 trafilatura 提正文，去掉导航/页脚

用法：
  python src/parse.py --raw data/raw --metadata data/metadata.csv --out data/docs.jsonl
"""

import argparse
import csv
import json
import os
import re


def clean(text):
    text = text.replace("\u3000", " ").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def pdf_text(path, min_chars_per_page=120):
    import fitz  # pymupdf
    doc = fitz.open(path)
    pages = [p.get_text("text") for p in doc]
    avg = sum(len(p) for p in pages) / max(1, len(pages))
    if avg < min_chars_per_page:
        raise ValueError(
            "%s 平均每页仅 %.0f 字，疑似扫描件，请换一份可复制文本的 PDF 或先 OCR"
            % (os.path.basename(path), avg))
    return clean("\n".join(pages)), len(pages)


# 只在这些关键词命中的页面上抽表格，445 页的本子全页抽会跑十几分钟
TABLE_HINT = ("课程计划", "教学计划", "开课", "课内实验")
TABLE_MARK = "【表格】"


def pdf_tables(path):
    """返回 [[表头..., 行...], ...]，抽取失败返回空列表。"""
    import pdfplumber
    out = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables() or []:
                rows = [[(c or "").strip() for c in row] for row in table if any(row)]
                if len(rows) >= 2:
                    out.append(rows)
    return out


def table_to_text(rows):
    header = rows[0]
    lines = [" | ".join(header)]
    for row in rows[1:]:
        if any(row):
            lines.append(" | ".join(c or "" for c in row))
    return "\n".join(lines)


def html_text(path):
    import trafilatura
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    body = trafilatura.extract(raw, include_tables=True) or ""
    return clean(body)


def pdf_doc(path, with_tables=True, min_chars_per_page=120):
    """整本 PDF -> 文本，表格就地插在该页正文之后（顺序不乱，切块时能定位到专业）。"""
    import fitz
    doc = fitz.open(path)
    pages = [p.get_text("text") for p in doc]
    doc.close()
    avg = sum(len(p) for p in pages) / max(1, len(pages))
    if avg < min_chars_per_page:
        raise ValueError(
            "%s 平均每页仅 %.0f 字，疑似扫描件，请换一份可复制文本的 PDF 或先 OCR"
            % (os.path.basename(path), avg))

    n_tables = 0
    if with_tables:
        hot = [i for i, t in enumerate(pages) if any(k in t for k in TABLE_HINT)]
        if hot:
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                for i in hot:
                    if i >= len(pdf.pages):
                        continue
                    try:
                        raw_tables = pdf.pages[i].extract_tables() or []
                    except Exception:
                        continue
                    blocks = []
                    for table in raw_tables:
                        rows = [[(c or "").strip() for c in row] for row in table if any(row)]
                        if len(rows) >= 2:
                            blocks.append(table_to_text(rows))
                    if blocks:
                        pages[i] = pages[i] + "\n\n" + TABLE_MARK + "\n" + "\n\n".join(blocks)
                        n_tables += len(blocks)
    return clean("\n".join(pages)), len(pages), n_tables


def load_metadata(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--metadata", default="data/metadata.csv")
    ap.add_argument("--out", default="data/docs.jsonl")
    ap.add_argument("--no-tables", action="store_true", help="跳过表格抽取，快很多")
    args = ap.parse_args()

    meta = {m["file_name"]: m for m in load_metadata(args.metadata)}
    written = skipped = 0
    with open(args.out, "w", encoding="utf-8") as fout:
        for name in sorted(os.listdir(args.raw)):
            path = os.path.join(args.raw, name)
            if not os.path.isfile(path):
                continue
            m = meta.get(name)
            if m is None:
                print("[skip] %s 未在 metadata.csv 登记" % name)
                skipped += 1
                continue
            try:
                pages = n_tables = 0
                if name.lower().endswith(".pdf"):
                    text, pages, n_tables = pdf_doc(path, with_tables=not args.no_tables)
                    print("[ok] %s  %d 页  %d 字  %d 张表"
                          % (name, pages, len(text), n_tables))
                elif name.lower().endswith((".html", ".htm")):
                    text = html_text(path)
                else:
                    print("[skip] 不支持 %s" % name)
                    skipped += 1
                    continue
            except Exception as e:
                print("[fail] %s: %s" % (name, e))
                skipped += 1
                continue

            doc = {
                "doc_id": m["doc_id"],
                "title": m["title"],
                "doc_type": m["doc_type"],
                "cohort": m.get("cohort", ""),
                "academic_year": m.get("academic_year", ""),
                "status": m.get("status", "active"),
                "version_policy": m.get("version_policy", "supersede"),
                "page_count": pages,
                "table_count": n_tables,
                "text": text,
            }
            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
            written += 1
    print("docs written:", written, "skipped:", skipped, "->", args.out)


if __name__ == "__main__":
    main()
