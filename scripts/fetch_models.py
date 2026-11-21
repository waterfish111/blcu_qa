"""下载 embedding / rerank 模型到本地缓存，并记录解析后的真实路径。

为什么要单独写这个脚本：
  1. 同一套模型在 ModelScope 和 HuggingFace 上的 ID 不一样，网络可达性也不一样；
  2. 下载位置要可复现（统一落在仓库内 models/，可用 BLCU_MODEL_CACHE 覆盖），
     不能靠每个脚本各自的默认值；
  3. 下完之后要能回答"这个模型到底落在哪个目录"——写进 models.lock.json，检索层据此加载。

用法：
    python scripts/fetch_models.py --which small   # 先跑通用：约 100MB + 1.1GB
    python scripts/fetch_models.py --which full    # 正式指标用：约 2.3GB + 2.3GB
    python scripts/fetch_models.py --which all
    python scripts/fetch_models.py --probe         # 只探测哪些 ID 能下载，不下载
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "models.lock.json"

# 统一缓存目录；可用 BLCU_MODEL_CACHE 指定别处
CACHE_DIR = Path(os.environ.get("BLCU_MODEL_CACHE") or (ROOT / "models"))

# 每个角色给一串候选 ID：先在 ModelScope 试，再退到 HuggingFace 镜像
MODELS = {
    "embedding": {
        "full": {
            "role": "embedding",
            "candidates": ["BAAI/bge-m3", "AI-ModelScope/bge-m3"],
        },
        "small": {
            "role": "embedding",
            "candidates": ["AI-ModelScope/bge-small-zh-v1.5", "BAAI/bge-small-zh-v1.5"],
        },
    },
    "rerank": {
        "full": {
            "role": "rerank",
            "candidates": ["BAAI/bge-reranker-v2-m3", "AI-ModelScope/bge-reranker-v2-m3"],
        },
        "small": {
            "role": "rerank",
            "candidates": ["AI-ModelScope/bge-reranker-base", "BAAI/bge-reranker-base"],
        },
    },
}

HF_MIRROR = "https://hf-mirror.com"

# 只取推理真正需要的文件：不少仓库同时放了 pytorch_model.bin / model.safetensors / onnx/model.onnx
# 三份等价权重，全下要多花 2–3 倍流量（bge-reranker-base 全量 3.3GB，只取 safetensors 是 1.1GB）。
ALLOW = ["*.json", "*.safetensors", "*.model", "*.txt", "*.spiece.model", "*.bpe.model", "1_Pooling/*"]


def _prepare_env():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # ModelScope 缓存
    os.environ.setdefault("MODELSCOPE_CACHE", str(CACHE_DIR))
    # HuggingFace 缓存（镜像路径下单独一层，避免和 modelscope 的目录混在一起）
    os.environ.setdefault("HF_HOME", str(CACHE_DIR / "hf"))
    os.environ.setdefault("HF_ENDPOINT", HF_MIRROR)


def _try_modelscope(model_id: str):
    try:
        from modelscope import snapshot_download  # type: ignore
    except Exception as e:  # 没装 modelscope 就跳过
        return None, f"modelscope 不可用: {e}"
    try:
        path = snapshot_download(model_id, cache_dir=str(CACHE_DIR), allow_patterns=ALLOW)
        return Path(path), None
    except Exception as e:
        return None, f"modelscope 失败: {str(e)[:120]}"


def _try_hf(model_id: str):
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception as e:
        return None, f"huggingface_hub 不可用: {e}"
    try:
        path = snapshot_download(repo_id=model_id, cache_dir=str(CACHE_DIR / "hf"),
                                 allow_patterns=ALLOW)
        return Path(path), None
    except Exception as e:
        return None, f"hf 失败: {str(e)[:120]}"


def fetch(entry: dict, probe: bool = False):
    for model_id in entry["candidates"]:
        for name, fn in (("modelscope", _try_modelscope), ("hf-mirror", _try_hf)):
            t0 = time.time()
            if probe and name == "hf-mirror":
                continue
            print(f"  -> {name}: {model_id} ...", end=" ", flush=True)
            path, err = fn(model_id)
            if path is not None:
                size_mb = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6
                print(f"OK {size_mb:.0f}MB {time.time() - t0:.0f}s")
                return {"id": model_id, "source": name, "path": str(path), "size_mb": round(size_mb)}
            print("失败", err)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["small", "full", "all"], default="small")
    ap.add_argument("--probe", action="store_true", help="只探测，不真正下载")
    args = ap.parse_args()

    _prepare_env()
    print(f"缓存目录: {CACHE_DIR}")
    print(f"HF_ENDPOINT: {os.environ['HF_ENDPOINT']}")

    picks = {"small": ["small"], "full": ["full"], "all": ["small", "full"]}[args.which]
    lock = json.loads(LOCK.read_text(encoding="utf-8")) if LOCK.exists() else {}

    for role in ("embedding", "rerank"):
        for size in picks:
            entry = MODELS[role][size]
            if args.probe:
                print(f"[探测] {role}/{size}")
            else:
                print(f"[下载] {role}/{size}")
            got = fetch(entry, probe=args.probe)
            if got:
                got["size"] = size
                lock[role] = got
                print(f"  {role} <- {got['path']}")
            else:
                print(f"  [warn] {role}/{size} 全部候选都失败，检索层会自动降级")

    if not args.probe and lock:
        LOCK.write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写入 {LOCK}")
        for role, v in lock.items():
            print(f"  {role}: {v['id']} ({v['source']}, {v['size_mb']}MB)")


if __name__ == "__main__":
    sys.exit(main())
