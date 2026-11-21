"""最小 HTTP 服务：把「检索 → 重排 → 生成 → 校验」包成一个 /ask 接口。

启动：
    python api.py                 # 默认 127.0.0.1:8000
    uvicorn api:app --reload      # 开发时自动重载

设计上只做三件事，前端拿到的东西要够画界面，但不替前端做决定：
  1. /ask 返回答案 + 拒答原因 + 每张出处卡（路径、分数、原文），前端直接渲染
  2. /health 返回"现在是什么状态"：多少块、有没有向量模型、阈值多少
  3. 检索器只在启动时建一次，模型懒加载；没装模型自动退化成纯 BM25

这个文件故意不做登录、不做历史记录、不做并发优化——演示用，不是生产系统。
"""

import json
import os
import sys
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import generate as gen          # noqa: E402
import retrieve as ret          # noqa: E402

CHUNKS_PATH = Path(os.environ.get("BLCU_CHUNKS", ROOT / "data" / "chunks.jsonl"))
THRESHOLD = float(os.environ.get("BLCU_THRESHOLD", "0.35"))
TOPK = int(os.environ.get("BLCU_TOPK", "5"))

app = FastAPI(title="教务通 · 培养方案问答", version="0.1.0",
              description="北京语言大学培养方案 RAG 演示服务")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # 演示用：Vite 开发服务器在 5173，需要跨域
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever = None
ready = {"chunks": 0, "dense": False, "build_seconds": None, "error": None}

# 演示时同一个问题常被反复问（面试官会点第二遍），把整条结果缓存住。
# 只缓存 32 条、按最近使用淘汰——不是为了性能，是为了"第二遍立刻出来"的观感。
_CACHE = {}
_CACHE_MAX = 32


def load_chunks(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@app.on_event("startup")
def startup():
    global retriever
    t0 = time.time()
    chunks = load_chunks(CHUNKS_PATH)
    retriever = ret.HybridRetriever(chunks)
    retriever.build(dense=True)     # 没模型会自动降级，不会抛
    ready["chunks"] = len(chunks)
    ready["dense"] = retriever.vectors is not None
    ready["build_seconds"] = round(time.time() - t0, 1)
    gen.load_dotenv()
    print("[api] 就绪：%d 块，向量=%s，耗时 %.1fs"
          % (ready["chunks"], ready["dense"], ready["build_seconds"]))

    if os.environ.get("BLCU_WARMUP") == "1":
        # 预热：第一次查询要现编向量、加载重排模型，预热掉省得演示时卡 10 秒
        t = time.time()
        retriever.search("数字经济专业准予毕业总学分", topk=TOPK)
        print("[api] 预热完成 %.1fs" % (time.time() - t))


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=200, description="学生的问题")
    cohort: str = Field("", description="限定年级，如 2024级；留空表示不限")
    topk: int = Field(TOPK, ge=1, le=20)


@app.get("/health")
def health():
    has_key = bool(os.environ.get("DEEPSEEK_API_KEY") or gen.load_dotenv())
    return {"ok": retriever is not None, "chunks": ready["chunks"],
            "dense": ready["dense"], "threshold": THRESHOLD,
            "build_seconds": ready["build_seconds"], "llm_ready": has_key}


@app.post("/ask")
def ask(req: AskRequest):
    if retriever is None:
        return JSONResponse(status_code=503, content={"error": "检索器还在初始化"})

    key = (req.question.strip(), req.cohort.strip(), req.topk)
    if key in _CACHE:
        _CACHE[key]["cached"] = True
        # 命中缓存时把耗时归零，前端会看到"立刻出来"
        hit_cached = dict(_CACHE[key])
        hit_cached["latency_ms"] = {"search": 0, "total": 0}
        return hit_cached

    t0 = time.time()
    hit = retriever.search(req.question, cohort=req.cohort.strip(), topk=req.topk)
    t_search = time.time() - t0

    res = gen.answer(req.question, hit["chunks"], hit["top_score"],
                     threshold=THRESHOLD, chunks_limit=req.topk)

    # 把 [n] 编号翻成前端要用的卡片：编号 → 块
    cards = {}
    for n, chunk in enumerate(hit["chunks"][:req.topk], 1):
        cards[n] = {
            "n": n,
            "chunk_id": chunk.get("chunk_id"),
            "doc_id": chunk.get("doc_id"),
            "title": chunk.get("title"),
            "path": chunk.get("section_path") or "（无路径）",
            "article_no": chunk.get("article_no") or "",
            "score": round(hit.get("scores", {}).get(chunk["chunk_id"], 0.0), 4),
            "text": chunk.get("text", ""),
        }

    used_ns = [int(x) for x in gen.CITE_RE.findall(res["answer"])]
    payload = {
        "answer": res["answer"],
        "abstained": res["abstained"],
        "reason": res.get("reason", ""),
        "citation_ok": res.get("citation_ok", False),
        "trimmed": res.get("trimmed", False),
        "top_score": round(hit["top_score"], 4),
        "threshold": THRESHOLD,
        "chunks": [cards[n] for n in sorted(cards)],
        "used": sorted({n for n in used_ns if n in cards}),
        "latency_ms": {"search": round(t_search * 1000),
                       "total": round((time.time() - t0) * 1000)},
        "cached": False,
    }
    _CACHE[key] = payload
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.pop(next(iter(_CACHE)))      # 先进先出，够用了
    return payload


@app.get("/")
def index():
    """演示页就在 web/ 里，不需要构建：python api.py 之后直接开这个地址。"""
    page = ROOT / "web" / "index.html"
    if page.exists():
        return FileResponse(page)
    return JSONResponse({"hint": "web/index.html 不见了",
                         "ask": "POST /ask  {\"question\": \"...\"}"})


if (ROOT / "web").exists():
    app.mount("/web", StaticFiles(directory=ROOT / "web"), name="web")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ.get("BLCU_HOST", "127.0.0.1"),
                port=int(os.environ.get("BLCU_PORT", "8000")))
