"""检索层：BM25 + dense 双路召回，RRF 融合，可选重排。

依赖（缺哪个就在对应分支里跳过，方便先跑通纯 BM25 基线）：
  jieba, rank-bm25, sentence-transformers

用法：
  from retrieve import HybridRetriever
  r = HybridRetriever(chunks)
  r.build()
  hits = r.search("大二下能不能选大三的课", cohort="2024级", topk=5)
"""

import json
import math
import re
from collections import Counter
from pathlib import Path

try:
    import jieba
except ImportError:  # 允许在没装依赖时导入
    jieba = None

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None


def tokenize(text):
    """分词：装了 jieba 就用 jieba，否则用汉字二元组。

    二元组（"数字""字经""经济"…）对中文检索来说是个意外靠谱的基线：
    不用装任何东西、不怕新词（专业名、课程名基本都能切到），代价是索引大一点。
    """
    if jieba is not None:
        return [t for t in jieba.lcut_for_search(text) if t.strip()]
    text = re.sub(r"\s+", "", text)
    return [text[i:i + 2] for i in range(len(text) - 1)] or [text]


def resolve_model(role, default):
    """优先加载本地已下载的模型目录（models.lock.json），否则回落到模型 ID。

    这样检索层不用关心模型是 ModelScope 下的还是 HuggingFace 下的：
    fetch_models.py 下完会把真实路径写进 models.lock.json，这里读出来直接用。
    """
    lock_path = Path(__file__).resolve().parents[1] / "models.lock.json"
    if lock_path.exists():
        try:
            info = json.loads(lock_path.read_text(encoding="utf-8")).get(role) or {}
            if info.get("path") and Path(info["path"]).exists():
                return info["path"]
        except Exception:
            pass
    return default


def index_text(chunk):
    """送进索引的文本 = 面包屑（专业 > 小节）+ 正文。

    不拼面包屑会吃大亏：培养方案里「六、毕业要求及基本框架」这段正文在几十个
    专业里几乎一字不差，只有面包屑里的专业名能把它们区分开。
    """
    return ((chunk.get("section_path") or "") + " \n " + chunk.get("text", ""))


class _BM25:
    """纯 Python 的 BM25（k1=1.2，b=0.75），给没装 rank-bm25 的机器兜底。

    接口和 rank_bm25.BM25Okapi 一致（只有 get_scores），方便两个实现互换。
    """

    def __init__(self, docs_tokens, k1=1.2, b=0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(t) for t in docs_tokens]
        self.lens = [len(t) for t in docs_tokens]
        self.N = max(len(docs_tokens), 1)
        self.avgdl = (sum(self.lens) / self.N) or 1.0
        df = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5))
                    for t, n in df.items()}
        self._all = list(self.tfs)

    def get_scores(self, q_tokens):
        scores = [0.0] * len(self.tfs)
        for t in set(q_tokens):
            idf = self.idf.get(t)
            if not idf:
                continue
            for i, tf in enumerate(self._all):
                f = tf.get(t)
                if not f:
                    continue
                norm = self.k1 * (1 - self.b + self.b * self.lens[i] / self.avgdl)
                scores[i] += idf * f * (self.k1 + 1) / (f + norm)
        return scores


def rrf_fuse(rank_lists, k=60, topk=20):
    """Reciprocal Rank Fusion：只看排名，不用分数，省掉归一化。"""
    score = {}
    for ranked in rank_lists:
        for r, cid in enumerate(ranked):
            score[cid] = score.get(cid, 0.0) + 1.0 / (k + r + 1)
    return [cid for cid, _ in sorted(score.items(), key=lambda x: -x[1])[:topk]]


class HybridRetriever:
    def __init__(self, chunks, embed_model="BAAI/bge-m3",
                 rerank_model="BAAI/bge-reranker-v2-m3"):
        self.chunks = chunks
        self.by_id = {c["chunk_id"]: c for c in chunks}
        self.embed_model_name = resolve_model("embedding", embed_model)
        self.rerank_model_name = resolve_model("rerank", rerank_model)
        self.bm25 = None
        self.vectors = None
        self._embedder = None
        self._reranker = None

    # ---- 过滤：培养方案按年级，通用规章看是否失效 ----
    def _allowed(self, chunk, cohort, include_superseded=False):
        if chunk.get("status") == "superseded" and not include_superseded:
            return False
        if chunk.get("cohort") and cohort and chunk["cohort"] != cohort:
            return False
        return True

    def _vector_cache_path(self):
        """向量缓存文件名里带上模型名，换模型就自动重新编码，不会读到旧向量。"""
        slug = re.sub(r"[^0-9A-Za-z]+", "_", Path(str(self.embed_model_name)).name)[-40:]
        return Path(__file__).resolve().parents[1] / "data" / f"vectors_{slug}.npz"

    def build(self, rebuild_vectors=False, dense=True):
        if BM25Okapi is not None:
            self.bm25 = BM25Okapi([tokenize(index_text(c)) for c in self.chunks])
        else:
            print("[warn] 没装 rank-bm25，改用内置 BM25 实现")
            self.bm25 = _BM25([tokenize(index_text(c)) for c in self.chunks])

        if not dense:
            return

        ids = [c["chunk_id"] for c in self.chunks]
        cache = self._vector_cache_path()
        if cache.exists() and not rebuild_vectors:
            try:
                import numpy as np
                data = np.load(cache, allow_pickle=False)
                if list(data["ids"]) == ids:  # 语料变了就重算，避免向量和块错位
                    self.vectors = data["vectors"]
                    print(f"[cache] 复用向量 {cache.name} {self.vectors.shape}")
                    return
                print("[cache] 语料已变化，重新编码")
            except Exception as e:
                print("[cache] 读取失败，重新编码:", e)

        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.embed_model_name)
            self.vectors = self._embedder.encode(
                [index_text(c) for c in self.chunks],
                normalize_embeddings=True, show_progress_bar=True, batch_size=32)
            try:
                import numpy as np
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.savez(cache, vectors=self.vectors.astype("float32"), ids=np.array(ids))
                print(f"[cache] 已写入向量 {cache} {self.vectors.shape}")
            except Exception as e:
                print("[cache] 写入失败（不影响本次检索）:", e)
        except Exception as e:  # 模型没下载/没装，降级为纯 BM25
            print("[warn] dense 检索不可用，降级为 BM25 only:", e)

    def _bm25_rank(self, query, cohort, topk):
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(self.chunks)), key=lambda i: -scores[i])
        return [self.chunks[i]["chunk_id"] for i in order
                if self._allowed(self.chunks[i], cohort)][:topk]

    def _get_embedder(self):
        """向量是从缓存读的，但查询本身还得现场编码，所以按需加载模型。"""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.embed_model_name)
        return self._embedder

    def _dense_rank(self, query, cohort, topk):
        if self.vectors is None:
            return []
        import numpy as np
        q = self._get_embedder().encode([query], normalize_embeddings=True)[0]
        sims = self.vectors @ q
        order = np.argsort(-sims)
        return [self.chunks[i]["chunk_id"] for i in order
                if self._allowed(self.chunks[i], cohort)][:topk]

    def _rerank(self, query, cand_ids, topk):
        """返回 (ids, 最高分, {chunk_id: 分数})。分数给前端画卡片用。"""
        try:
            from sentence_transformers import CrossEncoder
            if self._reranker is None:
                self._reranker = CrossEncoder(self.rerank_model_name)
            # 重排也要看到面包屑：跨届题的正文在不同专业里一字不差，
            # 只把 text 喂给 cross-encoder，模型根本分不清是哪一届哪个专业，
            # 实测就把 recall@5 从 0.625 拉低到 0.542。
            pairs = [(query, index_text(self.by_id[c])) for c in cand_ids]
            scores = self._reranker.predict(pairs)
            ranked = sorted(zip(cand_ids, scores), key=lambda x: -x[1])
            top = ranked[:topk]
            return ([c for c, _ in top],
                    float(top[0][1]) if top else 0.0,
                    {c: float(s) for c, s in ranked})
        except Exception:
            # 没有重排模型时，用融合顺序的结果和 1.0 兜底
            return cand_ids[:topk], 1.0, {}

    def search(self, query, cohort="", topk=5, bm25_topk=20, dense_topk=20,
               use_rerank=True, include_superseded=False):
        lists = []
        if self.bm25 is not None:
            lists.append(self._bm25_rank(query, cohort, bm25_topk))
        if self.vectors is not None:
            lists.append(self._dense_rank(query, cohort, dense_topk))
        fused = rrf_fuse(lists, topk=max(bm25_topk, dense_topk)) if lists else []
        if use_rerank and fused:
            ids, top_score, score_map = self._rerank(query, fused, topk)
        else:
            ids, top_score, score_map = fused[:topk], 1.0, {}
        chunks = []
        for cid in ids:
            item = dict(self.by_id[cid])
            item["score"] = score_map.get(cid, 0.0)
            chunks.append(item)
        return {"chunk_ids": ids, "top_score": top_score,
                "scores": score_map, "chunks": chunks}
