"""评测：recall@k / MRR / 误拒答率 / 应拒答准确率 / 过期条款误用率

只用标准库，方便在没装检索依赖的机器上先跑通流程。

预测文件 eval/results/pred_*.jsonl，每行：
  {"id": "q001", "answer": "...", "abstained": false,
   "retrieved_chunk_ids": ["doc#c001", ...], "used_chunk_ids": ["doc#c001", ...]}

用法：
  python src/eval.py --questions eval/questions.jsonl \
                     --pred eval/results/pred_baseline.jsonl \
                     --metadata data/metadata.csv \
                     --ks 1 5 10 --out eval/results/metrics_baseline.json
"""

import argparse
import csv
import json
import os
from collections import defaultdict


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_status(metadata_path):
    """doc_id -> status(active/superseded)；缺文件时返回空表。"""
    status = {}
    if not metadata_path or not os.path.exists(metadata_path):
        return status
    with open(metadata_path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            status[row["doc_id"]] = row.get("status", "active")
    return status


def doc_of(chunk_id):
    return chunk_id.split("#", 1)[0]


def evaluate(questions, preds, status, ks):
    pred_by_id = {p["id"]: p for p in preds}
    n = len(questions)
    hits = defaultdict(int)
    rr_sum = 0.0
    false_abstain = false_abstain_n = 0
    correct_abstain = must_abstain = 0
    stale_misuse = stale_n = 0
    missing = []

    for q in questions:
        p = pred_by_id.get(q["id"])
        if p is None:
            missing.append(q["id"])
            continue

        gold = set(q.get("gold_chunk_ids") or [])
        ranked = p.get("retrieved_chunk_ids") or []
        abstained = bool(p.get("abstained"))

        if gold:
            for k in ks:
                if gold & set(ranked[:k]):
                    hits[k] += 1
            rank = next((i + 1 for i, c in enumerate(ranked) if c in gold), None)
            if rank:
                rr_sum += 1.0 / rank

        if q.get("expect_abstain"):
            must_abstain += 1
            if abstained:
                correct_abstain += 1
        else:
            false_abstain_n += 1
            if abstained:
                false_abstain += 1

            used_docs = {doc_of(c) for c in (p.get("used_chunk_ids") or [])}
            if status and any(status.get(d) == "superseded" for d in used_docs):
                stale_misuse += 1
            stale_n += 1

    answerable = n - must_abstain
    n_gold = sum(1 for q in questions if q.get("gold_chunk_ids"))
    out = {
        "n_questions": n,
        "n_scored": n - len(missing),
        "missing_pred_ids": missing,
        "n_with_gold": sum(1 for q in questions if q.get("gold_chunk_ids")),
    }
    for k in ks:
        denom = sum(1 for q in questions if q.get("gold_chunk_ids")) or 1
        out["recall@%d" % k] = round(hits[k] / denom, 4)
    out["mrr"] = round(rr_sum / (n_gold or 1), 4)  # 分母是所有有金标的题，未命中记 0
    out["false_abstain_rate"] = round(false_abstain / (false_abstain_n or 1), 4)
    out["abstain_accuracy"] = round(correct_abstain / (must_abstain or 1), 4)
    out["stale_misuse_rate"] = round(stale_misuse / (stale_n or 1), 4) if status else None
    out["n_answerable"] = answerable
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="eval/questions.jsonl")
    ap.add_argument("--pred", required=True)
    ap.add_argument("--metadata", default="data/metadata.csv")
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10])
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    questions = load_jsonl(args.questions)
    preds = load_jsonl(args.pred)
    status = load_status(args.metadata)
    metrics = evaluate(questions, preds, status, args.ks)

    width = max(len(k) for k in metrics)
    for k, v in metrics.items():
        print("%-*s  %s" % (width, k, v))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print("saved ->", args.out)


if __name__ == "__main__":
    main()
