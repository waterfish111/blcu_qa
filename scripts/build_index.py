"""一步建索引：解析 -> 切块 -> 打印统计，为后续 BM25/dense 建库做准备。

  # 语料在桌面，不搬进仓库，直接指过去
  python scripts/build_index.py --raw "C:\\Users\\Lenovo\\Desktop\\培养方案"

  # 只想快跑（跳过表格抽取），或换切块策略做消融
  python scripts/build_index.py --raw ... --no-tables --mode window
"""

import argparse
import json
import os
import statistics
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import chunk as chunk_mod  # noqa: E402
import parse as parse_mod  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--metadata", default="data/metadata.csv")
    ap.add_argument("--docs", default="data/docs.jsonl")
    ap.add_argument("--out", default="data/chunks.jsonl")
    ap.add_argument("--max-chars", type=int, default=400)
    ap.add_argument("--overlap", type=int, default=80)
    ap.add_argument("--mode", default="auto", choices=["auto", "article", "major", "window"])
    ap.add_argument("--no-tables", action="store_true")
    args = ap.parse_args()

    sys.argv = ["parse.py", "--raw", args.raw, "--metadata", args.metadata, "--out", args.docs]
    if args.no_tables:
        sys.argv.append("--no-tables")
    parse_mod.main()

    if not os.path.exists(args.docs):
        print("没有解析出任何文档，检查 raw 目录与 metadata.csv 是否对得上")
        return

    total = 0
    per_doc = Counter()
    sizes = []
    no_path = 0
    with open(args.docs, "r", encoding="utf-8-sig") as fin, \
            open(args.out, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            doc = json.loads(line)
            cs = chunk_mod.chunk_doc(doc, args.max_chars, args.overlap, mode=args.mode)
            for c in cs:
                fout.write(json.dumps(c, ensure_ascii=False) + "\n")
                sizes.append(len(c["text"]))
                if not c["section_path"]:
                    no_path += 1
            per_doc[doc["doc_id"]] = len(cs)
            total += len(cs)

    print("\nchunks:", total, "| mode:", args.mode)
    for k, v in per_doc.most_common():
        print("  %-24s %d" % (k, v))
    if sizes:
        print("长度 中位 %d / 均值 %.0f / 最大 %d"
              % (statistics.median(sizes), statistics.mean(sizes), max(sizes)))
        print("无路径块: %d (%.1f%%)" % (no_path, 100.0 * no_path / total))
    print("->", args.out)


if __name__ == "__main__":
    main()
