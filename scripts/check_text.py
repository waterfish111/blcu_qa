"""D1 第一道关：抽查 PDF 是不是可读文本（不是扫描件、不是乱码）。

  python scripts/check_text.py --raw "C:\\Users\\Lenovo\\Desktop\\培养方案"

对每份 PDF 取第 1/10/50 页，统计中文字符占比并打印前 60 字。
判读标准：
  - 每页字符数 < 120        → 扫描件，别用它
  - 中文占比 < 0.30         → 编码有问题（缺 ToUnicode），抽出来是乱码
  - 以上都通过              → 可以直接进 build_index
"""

import argparse
import os
import sys

CJK = lambda ch: "\u4e00" <= ch <= "\u9fff"


def probe(path, pages=(0, 9, 49)):
    import fitz
    doc = fitz.open(path)
    rows = []
    for idx in pages:
        if idx >= doc.page_count:
            continue
        text = doc[idx].get_text("text").strip()
        cjk = sum(1 for ch in text if CJK(ch))
        ratio = cjk / max(1, len(text))
        rows.append((idx + 1, len(text), ratio, text[:60].replace("\n", " ")))
    doc.close()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    names = sorted(n for n in os.listdir(args.raw) if n.lower().endswith(".pdf"))
    if not names:
        print("没找到 PDF：", args.raw)
        return

    for name in names:
        path = os.path.join(args.raw, name)
        print("=" * 70)
        print(name)
        for page_no, chars, ratio, head in probe(path):
            if chars < 120:
                verdict = "疑似扫描件"
            elif ratio < 0.30:
                verdict = "中文占比低，可能乱码"
            else:
                verdict = "OK"
            print("  p%-4d %5d 字  中文 %4.0f%%  %-16s %s"
                  % (page_no, chars, ratio * 100, verdict, head))


if __name__ == "__main__":
    main()
