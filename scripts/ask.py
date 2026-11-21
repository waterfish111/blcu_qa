"""问一个问题，把整条链路每一步都打印出来。

    $env:DEEPSEEK_API_KEY='sk-...'
    python scripts/ask.py "2024 版汉语言文学专业准予毕业课程学分是多少？" --cohort 2024级

和 run_eval.py 的区别：run_eval 跑批、只记数字；ask 跑一条、把"检索到什么、
模型怎么答、引用校验过没过"全摊开，用来人工验收和录演示视频。

  --no-generate   只看检索结果，不花钱
  --show-chunks   除了答案，把 5 张卡片的前 160 字也打出来
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import generate as gen  # noqa: E402
import retrieve as ret  # noqa: E402


def load_jsonl(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--chunks", default="data/chunks.jsonl")
    ap.add_argument("--cohort", default="", help="年级过滤，如 2024级；留空不过滤")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--no-generate", action="store_true")
    ap.add_argument("--no-dense", action="store_true")
    ap.add_argument("--show-chunks", action="store_true")
    args = ap.parse_args()

    chunks = load_jsonl(args.chunks)
    r = ret.HybridRetriever(chunks)
    r.build(dense=not args.no_dense)

    print("=" * 70)
    print("问题：", args.question)
    print("年级过滤：", args.cohort or "（不过滤）")
    print("=" * 70)

    hit = r.search(args.question, cohort=args.cohort, topk=args.topk)
    print("\n[1] 检索到 %d 张卡片，最高重排分 %.3f（< %.2f 视为无依据）"
          % (len(hit["chunks"]), hit["top_score"], args.threshold))
    for i, c in enumerate(hit["chunks"], 1):
        print("    [%d] %s#%s  %s" % (i, c["doc_id"], c["chunk_id"].split("#")[-1],
                                      c.get("section_path", "") or "（无路径）"))
        if args.show_chunks:
            print("        " + c["text"][:160].replace("\n", " "))

    if args.no_generate:
        return
    gen.load_dotenv()
    if not os.environ.get("DEEPSEEK_API_KEY"):
        sys.exit("\n缺少 DEEPSEEK_API_KEY（写进 blcu-qa\\.env 即可），"
                 "加 --no-generate 可只看检索。")

    res = gen.answer(args.question, hit["chunks"], hit["top_score"],
                     threshold=args.threshold, debug=True)
    print("\n[2] 模型回答：\n" + res["answer"])
    print("\n[3] 校验：abstained=%s citation_ok=%s 引用了 %d 张卡片"
          % (res["abstained"], res["citation_ok"], len(res["used_chunk_ids"])))
    for cid in res["used_chunk_ids"]:
        c = r.by_id.get(cid, {})
        print("    ✅ %s  %s" % (cid, c.get("section_path", "")))
    if res.get("debug"):
        print("    ⚠ 未通过校验的引用：", res["debug"]["bad_citations"])


if __name__ == "__main__":
    main()
