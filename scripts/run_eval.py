"""跑评测集：对每条问题做检索 + 生成，写出 pred 文件，再调 eval.py 算指标。

  export DEEPSEEK_API_KEY=...
  python scripts/run_eval.py --chunks data/chunks.jsonl --cohort 2024级 \
        --pred eval/results/pred_bm25_rerank.jsonl
  python src/eval.py --pred eval/results/pred_bm25_rerank.jsonl --out eval/results/metrics_bm25_rerank.json

只想看检索指标、不想花 API 额度时，加 --no-generate。
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
    ap.add_argument("--chunks", default="data/chunks.jsonl")
    ap.add_argument("--questions", default="eval/questions.jsonl")
    ap.add_argument("--cohort", default="",
                    help="题目没写 cohort 时的默认年级；留空表示不过滤")
    ap.add_argument("--pred", default="eval/results/pred.jsonl")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--no-dense", action="store_true",
                    help="不加载向量模型，只跑 BM25（拿对照基线用）")
    ap.add_argument("--no-generate", action="store_true")
    ap.add_argument("--limit", type=int, default=0,
                    help="只跑前 N 题（先花 2 毛钱试通流程用）")
    ap.add_argument("--debug", action="store_true",
                    help="把校验失败时的草稿与不合格引用编号写进 pred，方便查为什么被拒答")
    args = ap.parse_args()

    if not args.no_generate:
        gen.load_dotenv()
        if not os.environ.get("DEEPSEEK_API_KEY"):
            sys.exit("缺少 DEEPSEEK_API_KEY。把 key 写进 blcu-qa\\.env（一行 "
                     "DEEPSEEK_API_KEY=sk-...），或者设环境变量；也可以加 "
                     "--no-generate 只测检索。")

    chunks = load_jsonl(args.chunks)
    questions = load_jsonl(args.questions)
    if args.limit:
        questions = questions[:args.limit]
    print("chunks:", len(chunks), "questions:", len(questions))

    r = ret.HybridRetriever(chunks)
    r.build(dense=not args.no_dense)

    os.makedirs(os.path.dirname(args.pred) or ".", exist_ok=True)
    with open(args.pred, "w", encoding="utf-8") as fout:
        for q in questions:
            cohort = q["cohort"] if "cohort" in q else args.cohort
            hit = r.search(q["q"], cohort=cohort, topk=args.topk,
                           use_rerank=not args.no_rerank)
            row = {"id": q["id"], "q": q["q"],
                   "retrieved_chunk_ids": hit["chunk_ids"],
                   "top_score": hit["top_score"]}
            if args.no_generate:
                row.update({"answer": "", "abstained": False, "used_chunk_ids": [],
                            "citation_ok": None})
            else:
                res = gen.answer(q["q"], hit["chunks"], hit["top_score"],
                                 threshold=args.threshold, debug=args.debug)
                row.update({"answer": res["answer"],
                            "abstained": res["abstained"],
                            "reason": res.get("reason", ""),
                            "used_chunk_ids": res["used_chunk_ids"],
                            "citation_ok": res["citation_ok"]})
                if args.debug and res.get("debug"):
                    row["debug"] = res["debug"]
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            print("  %-6s %s" % (q["id"], "abstain" if row["abstained"] else "answered"))
    print("->", args.pred)


if __name__ == "__main__":
    main()
