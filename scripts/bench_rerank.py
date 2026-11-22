# -*- coding: utf-8 -*-
"""量一下重排模型：fp32 和 int8 动态量化，四项数字对比。

用法（在仓库根执行）：
    python scripts/bench_rerank.py --mode both
    python scripts/bench_rerank.py --mode int8 --save

四项：加载后内存（RSS 增量）、单次 40 对的耗时、打分是否一致、磁盘占用。
打分一致性用两个指标看：
    - 最大绝对误差（logit 差多少）
    - 排序一致率（同一批候选，两种模型的 top-k 顺序重合比例）
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import psutil  # noqa: E402
import torch  # noqa: E402
from sentence_transformers import CrossEncoder  # noqa: E402

from retrieve import resolve_model  # noqa: E402

INT8_DIR = ROOT / "models" / "rerank_int8"


def rss_mb():
    return psutil.Process().memory_info().rss / 1024 / 1024


def load_pairs(n_pairs, n_queries=8):
    """从真实语料里拼 (query, passage) 对，长度分布跟线上一致。"""
    chunks = []
    with open(ROOT / "data" / "chunks.jsonl", encoding="utf-8-sig") as f:
        for i, line in enumerate(f):
            if i % 97 == 0:  # 抽样，别把 6129 条全读进内存
                chunks.append(json.loads(line))
            if len(chunks) >= n_pairs:
                break
    questions = []
    with open(ROOT / "eval" / "questions.jsonl", encoding="utf-8-sig") as f:
        for line in f:
            q = json.loads(line)
            if not q.get("expect_abstain"):
                questions.append(q["q"])
    pairs = [(questions[i % len(questions)], c["text"][:900]) for i, c in enumerate(chunks)]
    return pairs


def measure(ce, pairs, repeat=3):
    times = []
    with torch.no_grad():
        for _ in range(repeat):
            t0 = time.time()
            scores = ce.predict(pairs)
            times.append(time.time() - t0)
    return times, scores


def quantize(ce):
    """把 HF 模型里的 Linear 层换成 int8 动态量化版本（权重压 4 倍，激活运行时量化）。

    必须 inplace=True：CrossEncoder 内部把这个模型同时挂成属性 `model` 和 Sequential 的
    第 0 个子模块，非原地转换会返回一个新对象，旧模块仍留在 Sequential 里，前向就会走错的路径。
    """
    torch.ao.quantization.quantize_dynamic(
        ce.model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True
    )
    import gc
    gc.collect()  # 原来那份 fp32 权重要等回收，不然内存数字会虚高
    return ce


def param_report(model):
    """参数都花在哪：embedding 占大头时，int8 能省的只有 Linear，磁盘不会掉那么多。"""
    emb = sum(p.numel() for p in model.get_input_embeddings().parameters())
    lin = sum(p.numel() for n, p in model.named_parameters() if n.endswith("weight") and p.dim() == 2)
    total = sum(p.numel() for p in model.parameters())
    return {
        "total_m": round(total / 1e6, 1),
        "embedding_m": round(emb / 1e6, 1),
        "linear_m": round(lin / 1e6, 1),
        "other_m": round((total - emb) / 1e6, 1),
    }


def disk_mb(path):
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / 1024 / 1024
    if not path.exists():
        return 0.0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1024 / 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fp32", "int8", "both"], default="both")
    ap.add_argument("--model", default=resolve_model("rerank", "BAAI/bge-reranker-base"))
    ap.add_argument("--pairs", type=int, default=40)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--save", action="store_true", help="把 int8 权重存到 models/rerank_int8")
    args = ap.parse_args()

    base_rss = rss_mb()
    t0 = time.time()
    ce = CrossEncoder(str(args.model), max_length=512)
    load_s = time.time() - t0
    fp32_rss = rss_mb()
    print(f"[fp32] 加载 {load_s:.1f}s  RSS {fp32_rss - base_rss:.0f} MB", flush=True)

    pairs = load_pairs(args.pairs)
    print(f"[data] {len(pairs)} 对，平均 {statistics.mean(len(p) for _, p in pairs):.0f} 字", flush=True)

    fp32_times, fp32_scores = measure(ce, pairs, args.repeat)
    fp32_lat = statistics.median(fp32_times)
    print(f"[fp32] 40 对 {fp32_lat:.2f}s（{fp32_lat / len(pairs) * 1000:.0f} ms/对）", flush=True)

    result = {
        "pairs": len(pairs),
        "params": param_report(ce.model),
        "fp32": {
            "load_seconds": round(load_s, 1),
            "rss_mb": round(fp32_rss - base_rss),
            "latency_40": round(fp32_lat, 2),
            "ms_per_pair": round(fp32_lat / len(pairs) * 1000),
            "disk_mb": round(disk_mb(Path(args.model) / "model.safetensors"), 1),
        },
    }

    if args.mode == "fp32":
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    t0 = time.time()
    quantize(ce)
    quant_s = time.time() - t0
    int8_rss = rss_mb()
    print(f"[int8] 量化 {quant_s:.1f}s  RSS {int8_rss - base_rss:.0f} MB（增量 {int8_rss - fp32_rss:+.0f}）", flush=True)

    int8_times, int8_scores = measure(ce, pairs, args.repeat)
    int8_lat = statistics.median(int8_times)
    print(f"[int8] 40 对 {int8_lat:.2f}s（{int8_lat / len(pairs) * 1000:.0f} ms/对）", flush=True)

    diff = float(abs(torch.tensor(fp32_scores) - torch.tensor(int8_scores)).max())
    # 排序一致率：两种模型各自 top-5 的下标集合重合度
    k = min(5, len(pairs))
    fp32_top = set(torch.tensor(fp32_scores).topk(k).indices.tolist())
    int8_top = set(torch.tensor(int8_scores).topk(k).indices.tolist())
    agree = len(fp32_top & int8_top) / k
    print(f"[diff] logit 最大绝对误差 {diff:.3f}；top-{k} 排序一致率 {agree:.0%}", flush=True)

    size_mb = 0.0
    if args.save:
        INT8_DIR.mkdir(parents=True, exist_ok=True)
        out = INT8_DIR / "rerank_int8.pt"
        torch.save(
            {
                "state_dict": ce.model.state_dict(),
                "source": str(args.model),
                "engine": torch.backends.quantized.engine,
                "max_length": 512,
            },
            out,
        )
        size_mb = out.stat().st_size / 1024 / 1024
        print(f"[save] {out}  {size_mb:.0f} MB", flush=True)

    result["int8"] = {
        "quantize_seconds": round(quant_s, 1),
        "rss_mb": round(int8_rss - base_rss),
        "latency_40": round(int8_lat, 2),
        "ms_per_pair": round(int8_lat / len(pairs) * 1000),
        "speedup": round(fp32_lat / int8_lat, 2),
        "max_logit_diff": round(diff, 3),
        "top5_agreement": round(agree, 2),
        "saved_mb": round(size_mb, 1),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
