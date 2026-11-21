"""语料体检：每份文档的行碎片率、切块路径覆盖率、长度分布。

跑在 docs.jsonl + chunks.jsonl 上，只用标准库，几秒出结果。
README 的「语料质量」表就是这张表的输出。

  python scripts/quality_report.py --docs data/docs.jsonl --chunks data/chunks.jsonl
"""

import argparse
import collections
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import chunk as chunk_mod  # noqa: E402


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def fragmentation(text):
    """短行占比：PDF 文本层若是按列输出，正文会被拆成一字一行，这个值会飙高。"""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return 0.0
    return sum(1 for l in lines if len(l) <= 4) / float(len(lines))


def report(docs, chunks):
    by_doc = collections.defaultdict(list)
    for c in chunks:
        by_doc[c["doc_id"]].append(c)

    print("%-22s %6s %9s %6s %6s %6s %7s %s" % (
        "doc_id", "页", "字数", "碎片%", "块数", "无路径%", "中位长", "热门路径"))
    print("-" * 108)
    for d in docs:
        cs = by_doc.get(d["doc_id"], [])
        lens = [len(c["text"]) for c in cs] or [0]
        empty = sum(1 for c in cs if not c["section_path"])
        top = collections.Counter(c["section_path"] for c in cs if c["section_path"]).most_common(1)
        print("%-22s %6s %9s %5.1f%% %6d %5.1f%% %7d  %s" % (
            d["doc_id"],
            d.get("page_count", d.get("pages", "-")),
            d.get("char_count", len(d.get("text", ""))),
            100 * fragmentation(d.get("text", "")),
            len(cs),
            100.0 * empty / max(1, len(cs)),
            int(statistics.median(lens)),
            (top[0][0][:34] if top else "-"),
        ))

    total = len(chunks)
    empty = sum(1 for c in chunks if not c["section_path"])
    print("-" * 108)
    print("合计 %d 块，无路径 %d 块（%.1f%%）" % (total, empty, 100.0 * empty / max(1, total)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="data/docs.jsonl")
    ap.add_argument("--chunks", default="data/chunks.jsonl")
    ap.add_argument("--rechunk", action="store_true",
                    help="忽略 chunks.jsonl，用 src/chunk.py 现算一遍（改完切块逻辑后用）")
    args = ap.parse_args()

    docs = load_jsonl(args.docs)
    if args.rechunk or not os.path.exists(args.chunks):
        chunks = []
        for d in docs:
            chunks.extend(chunk_mod.chunk_doc(d))
    else:
        chunks = load_jsonl(args.chunks)
    report(docs, chunks)


if __name__ == "__main__":
    main()
