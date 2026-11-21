"""扫描语料目录，按文件名推断 doc_id / cohort / 版本语义，生成 metadata.csv。

培养方案是「按届适用」的：2024 级永远按 2024 版执行，2026 版不会取代它，
所以 version_policy=cohort，status 一律 active。学生手册这类才是 supersede。

用法：
  python scripts/scan_raw.py --raw data/raw --out data/metadata.csv
  python scripts/scan_raw.py --raw data/raw --out data/metadata.csv --keep-existing
"""

import argparse
import csv
import hashlib
import os
import re

FIELDS = ["doc_id", "title", "doc_type", "cohort", "academic_year",
          "effective_from", "effective_to", "version_policy", "status",
          "file_name", "source_url", "file_hash"]

YEAR_RE = re.compile(r"(20\d{2})\s*年?版?")
VOL_RE = re.compile(r"[（(]\s*(上|下|中)册\s*[)）]")

# 文件名里出现这些词 → 按「新版取代旧版」处理；否则一律按届适用
SUPERSEDE_HINTS = ("学生手册", "管理规定", "实施办法", "管理办法", "通知", "细则")


def guess(file_name):
    stem, ext = os.path.splitext(file_name)
    years = YEAR_RE.findall(stem)
    year = years[0] if years else ""
    vol = VOL_RE.search(stem)
    vol_tag = vol.group(1) if vol else ""

    doc_type = "培养方案" if "培养方案" in stem else "教务文件"
    policy = "supersede" if any(h in stem for h in SUPERSEDE_HINTS) else "cohort"

    if doc_type == "培养方案" and year:
        doc_id = "blcu_plan_%s" % year
        cohort = "%s级" % year
        title = "%s级本科人才培养方案%s" % (year, "（%s册）" % vol_tag if vol_tag else "")
    else:
        doc_id = "blcu_%s" % re.sub(r"[^\w]+", "_", stem)[:40]
        cohort = ""
        title = stem

    # 同一届分成上/下册的，用卷号区分，别让 doc_id 撞车
    if vol_tag:
        doc_id += "_vol%d" % {"上": 1, "中": 2, "下": 2}[vol_tag]

    return {
        "doc_id": doc_id,
        "title": title,
        "doc_type": doc_type,
        "cohort": cohort,
        "academic_year": year,
        "effective_from": "%s-09-01" % year if year else "",
        "effective_to": "",
        "version_policy": policy,
        "status": "active",
        "file_name": file_name,
        "source_url": "",
        "file_hash": "",
    }


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/metadata.csv")
    ap.add_argument("--keep-existing", action="store_true",
                    help="保留 csv 里已登记的行，只补新文件")
    ap.add_argument("--no-hash", action="store_true")
    args = ap.parse_args()

    existing = {}
    if args.keep_existing and os.path.exists(args.out):
        with open(args.out, "r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                existing[row["file_name"]] = row

    names = sorted(n for n in os.listdir(args.raw)
                   if os.path.isfile(os.path.join(args.raw, n))
                   and n.lower().endswith((".pdf", ".html", ".htm")))
    rows = []
    for name in names:
        row = existing.get(name) or guess(name)
        if not args.no_hash and not row.get("file_hash"):
            row["file_hash"] = sha256(os.path.join(args.raw, name))
        rows.append(row)

    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["cohort"], r["doc_id"])):
            writer.writerow({k: row.get(k, "") for k in FIELDS})
    print("registered %d files -> %s" % (len(rows), args.out))
    for row in rows:
        print("  %-22s %-8s %-14s %s" % (row["doc_id"], row["cohort"] or "-",
                                          row["version_policy"], row["file_name"]))


if __name__ == "__main__":
    main()
