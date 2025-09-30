# server.py
import os, time, pickle, requests
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, Range, MatchValue, SparseVector
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# ---------- Config ----------
COLLECTION = os.getenv("QDRANT_COLLECTION", "recipes")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/e5-base-v2")
TFIDF_PATH = os.getenv("TFIDF_PATH", "/app_state/tfidf.pkl")

# ---------- FastAPI ----------
app = FastAPI(title="Recipe Search (Qdrant)", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def index():
    return FileResponse("static/index.html")

# ---------- Startup: wait for Qdrant ready ----------
def wait_for_ready(url: str, timeout: int = 120, interval: float = 1.5):
    deadline = time.time() + timeout
    readyz = url.rstrip("/") + "/readyz"
    while time.time() < deadline:
        try:
            r = requests.get(readyz, timeout=3)
            if r.ok:
                return
        except Exception:
            pass
        time.sleep(interval)
    raise RuntimeError(f"Qdrant not ready at {readyz} after {timeout}s")

@app.on_event("startup")
def _startup():
    wait_for_ready(QDRANT_URL)
    global client, embedder, tfidf
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    embedder = SentenceTransformer(EMBED_MODEL)
    tfidf = None
    if os.path.exists(TFIDF_PATH):
        with open(TFIDF_PATH, "rb") as f:
            tfidf = pickle.load(f)

def sparse_enabled() -> bool:
    info = client.get_collection(COLLECTION)
    return bool(getattr(getattr(info, "config", None), "sparse_vectors", None))

# ---------- Search helpers ----------
def embed_query(text: str) -> List[float]:
    v = embedder.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
    return v.astype("float32").tolist()

def make_filter(cal_min: float | None, cal_max: float | None,
                min_protein: float | None,
                cuisine: str | None, diet: str | None) -> Filter | None:
    must = []
    if cal_min is not None or cal_max is not None:
        must.append(FieldCondition(key="macros_per_serving.cal", range=Range(gte=cal_min, lte=cal_max)))
    if min_protein is not None:
        must.append(FieldCondition(key="macros_per_serving.protein_g", range=Range(gte=min_protein)))
    if cuisine:
        must.append(FieldCondition(key="cuisines", match=MatchValue(value=cuisine)))
    if diet:
        must.append(FieldCondition(key="diet_tags", match=MatchValue(value=diet)))
    return Filter(must=must) if must else None

def to_sparse_vector(text: str) -> SparseVector:
    if tfidf is None:
        from sklearn.feature_extraction.text import TfidfVectorizer as _TV
        X = _TV(ngram_range=(1,2), min_df=1).fit_transform([text]).tocoo()
    else:
        X = tfidf.transform([text]).tocoo()
    return SparseVector(indices=X.col.tolist(), values=X.data.astype(float).tolist())

def rrf_fuse(results_lists, k=60):
    agg: Dict[Any, float] = {}
    for res in results_lists:
        for rank, h in enumerate(res, start=1):
            agg[h.id] = agg.get(h.id, 0.0) + 1.0/(k+rank)
    return sorted(agg.items(), key=lambda x: x[1], reverse=True)

# ---------- API ----------
@app.post("/search")
def search(
    query: str = Body(..., embed=True),
    limit: int = Body(5),
    cal_min: float | None = Body(None),
    cal_max: float | None = Body(None),
    min_protein: float | None = Body(None),
    cuisine: str | None = Body(None),
    diet: str | None = Body(None),
    hybrid: bool = Body(False)
):
    flt = make_filter(cal_min, cal_max, min_protein, cuisine, diet)

    # Dense
    qvec = embed_query(query)
    d_hits = client.search(COLLECTION, query_vector=("text_embedding", qvec), query_filter=flt, limit=max(50, limit), with_payload=True)

    # Hybrid (only if collection has sparse + tfidf exists)
    use_sparse = hybrid and sparse_enabled() and (tfidf is not None)
    if use_sparse:
        s_hits = client.search(COLLECTION, query_vector=("sparse_embedding", to_sparse_vector(query)), query_filter=flt, limit=max(50, limit), with_payload=True)
        fused = rrf_fuse([d_hits, s_hits])
        top_ids = [pid for pid,_ in fused[:limit]]
        recs = client.retrieve(COLLECTION, ids=top_ids, with_payload=True)
        order = {pid:i for i,(pid,_) in enumerate(fused)}
        recs.sort(key=lambda r: order.get(r.id, 1e9))
        results = [r.payload for r in recs]
    else:
        results = [h.payload for h in d_hits[:limit]]

    return {"hybrid_used": use_sparse, "count": len(results), "results": results}
