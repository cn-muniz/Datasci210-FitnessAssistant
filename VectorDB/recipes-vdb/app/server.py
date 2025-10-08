import os, time, pickle, threading
from typing import Any, Dict, List

from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Filter, FieldCondition, Range, MatchValue,
    SparseVector, ScoredPoint
)
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# ---------------- Config ----------------

COLLECTION = os.getenv("QDRANT_COLLECTION", "recipes")

def _from_env_or_file(var: str, default=None):
    path = os.getenv(f"{var}_FILE")
    if path and os.path.exists(path):
        return open(path, "r", encoding="utf-8").read().strip()
    return os.getenv(var, default)

QDRANT_URL = os.getenv("QDRANT_URL", "https://YOUR-CLOUD-URL:6333")
QDRANT_API_KEY = _from_env_or_file("QDRANT_API_KEY", None)

EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/e5-base-v2")
TFIDF_PATH = os.getenv("TFIDF_PATH", "/app_state/tfidf.pkl")

# --------------- App setup ---------------

app = FastAPI(title="Recipe Search (Qdrant Cloud)", version="0.5.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def index():
    return FileResponse("static/index.html")

# -------------- Global state --------------

STATE: Dict[str, Any] = {
    "client_ok": False,
    "collection_ok": False,
    "sparse_enabled": None,
    "model_loaded": False,
    "tfidf_loaded": False,
    "error": None,
}

client: QdrantClient | None = None
embedder: SentenceTransformer | None = None
tfidf: TfidfVectorizer | None = None

def _init_worker():
    """Initialize Qdrant client + model in background so the UI loads instantly."""
    global client, embedder, tfidf
    try:
        c = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)
        STATE["client_ok"] = True

        coll = c.get_collection(COLLECTION)
        STATE["collection_ok"] = True

        # Determine if sparse vectors are configured on this collection
        sparse_enabled = False
        try:
            cfg = coll.config
            # present when configured as named sparse vectors
            sparse_enabled = bool(getattr(cfg, "sparse_vectors", None))
            if not sparse_enabled:
                # fallback: some client versions tuck this under params.*
                params = getattr(cfg, "params", None)
                sparse_enabled = bool(getattr(params, "sparse_vectors", None))
        except Exception:
            sparse_enabled = False
        STATE["sparse_enabled"] = sparse_enabled

        eb = SentenceTransformer(EMBED_MODEL)
        STATE["model_loaded"] = True

        vec = None
        if os.path.exists(TFIDF_PATH):
            with open(TFIDF_PATH, "rb") as f:
                vec = pickle.load(f)
            STATE["tfidf_loaded"] = True

        # commit to globals after success
        client = c
        embedder = eb
        tfidf = vec
        STATE["error"] = None

    except Exception as e:
        STATE["error"] = f"{type(e).__name__}: {e}"

@app.on_event("startup")
def _startup():
    t = threading.Thread(target=_init_worker, daemon=True)
    t.start()

# --------------- Helpers -----------------

def embed_query(text: str) -> List[float]:
    if embedder is None:
        raise RuntimeError("Model not loaded yet")
    v = embedder.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
    return v.astype("float32").tolist()

def _add_range(must_list, key: str, min_v: float | None, max_v: float | None):
    if min_v is None and max_v is None:
        return
    must_list.append(FieldCondition(key=key, range=Range(gte=min_v, lte=max_v)))

def make_filter(
    cal_min: float | None, cal_max: float | None,
    protein_min: float | None, protein_max: float | None,
    carbs_min: float | None, carbs_max: float | None,
    fat_min: float | None, fat_max: float | None,
    sodium_min: float | None, sodium_max: float | None,
    cuisine_tag: str | None, diet_tag: str | None, meal_tag: str | None,
    vegan: bool | None, vegetarian: bool | None, pescatarian: bool | None,
    gluten_free: bool | None, dairy_free: bool | None,
) -> Filter | None:
    must = []
    # numeric ranges
    _add_range(must, "macros_per_serving.cal",       cal_min,     cal_max)
    _add_range(must, "macros_per_serving.protein_g", protein_min, protein_max)
    _add_range(must, "macros_per_serving.carbs_g",   carbs_min,   carbs_max)
    _add_range(must, "macros_per_serving.fat_g",     fat_min,     fat_max)
    _add_range(must, "macros_per_serving.sodium_mg", sodium_min,  sodium_max)

    # tag filters
    if cuisine_tag:
        must.append(FieldCondition(key="cuisines", match=MatchValue(value=cuisine_tag)))
    if diet_tag:
        must.append(FieldCondition(key="diet_tags", match=MatchValue(value=diet_tag)))
    if meal_tag:
        must.append(FieldCondition(key="course",   match=MatchValue(value=meal_tag)))

    # boolean diet flags (nested under diet_flags.*)
    def bmatch(flag_key: str, flag_val: bool | None):
        if flag_val is not None:
            must.append(FieldCondition(key=f"diet_flags.{flag_key}", match=MatchValue(value=bool(flag_val))))
    bmatch("vegan",         vegan)
    bmatch("vegetarian",    vegetarian)
    bmatch("pescatarian",   pescatarian)
    bmatch("gluten_free",   gluten_free)
    bmatch("dairy_free",    dairy_free)

    return Filter(must=must) if must else None

def to_sparse_vector(text: str) -> SparseVector:
    # If TF-IDF not preloaded, fit a tiny one-off vectorizer (works for simple hybrid demos)
    if tfidf is None:
        from sklearn.feature_extraction.text import TfidfVectorizer as _TV
        X = _TV(ngram_range=(1,2), min_df=1).fit_transform([text]).tocoo()
    else:
        X = tfidf.transform([text]).tocoo()
    return SparseVector(indices=X.col.tolist(), values=X.data.astype(float).tolist())

def rrf_fuse(results_lists: List[List[ScoredPoint]], k: int = 60):
    agg: Dict[Any, float] = {}
    for res in results_lists:
        for rank, h in enumerate(res, start=1):
            agg[h.id] = agg.get(h.id, 0.0) + 1.0 / (k + rank)
    return sorted(agg.items(), key=lambda x: x[1], reverse=True)

# --------------- Endpoints ----------------

@app.get("/status")
def status():
    ready_dense = bool(STATE["client_ok"] and STATE["collection_ok"] and STATE["model_loaded"])
    ready_hybrid = bool(ready_dense and STATE["sparse_enabled"])
    return {
        "collection": COLLECTION,
        "client_ok": STATE["client_ok"],
        "collection_ok": STATE["collection_ok"],
        "sparse_enabled": STATE["sparse_enabled"],
        "model_loaded": STATE["model_loaded"],
        "tfidf_loaded": STATE["tfidf_loaded"],
        "ready_for_dense": ready_dense,
        "ready_for_hybrid": ready_hybrid,
        "error": STATE["error"],
    }

@app.post("/search")
def search(
    query: str = Body(..., embed=True),
    limit: int = Body(5),

    cal_min: float | None = Body(None),
    cal_max: float | None = Body(None),

    protein_min: float | None = Body(None),
    protein_max: float | None = Body(None),

    carbs_min: float | None = Body(None),
    carbs_max: float | None = Body(None),

    fat_min: float | None = Body(None),
    fat_max: float | None = Body(None),

    sodium_min: float | None = Body(None),
    sodium_max: float | None = Body(None),

    cuisine_tag: str | None = Body(None),
    diet_tag: str | None = Body(None),
    meal_tag: str | None = Body(None),

    vegan: bool | None = Body(None),
    vegetarian: bool | None = Body(None),
    pescatarian: bool | None = Body(None),
    gluten_free: bool | None = Body(None),
    dairy_free: bool | None = Body(None),

    hybrid: bool = Body(False)
):
    if not (STATE["client_ok"] and STATE["collection_ok"] and STATE["model_loaded"]):
        raise HTTPException(status_code=503, detail={"message": "Server not ready", "status": STATE})

    flt = make_filter(
        cal_min, cal_max,
        protein_min, protein_max,
        carbs_min, carbs_max,
        fat_min, fat_max,
        sodium_min, sodium_max,
        cuisine_tag, diet_tag, meal_tag,
        vegan, vegetarian, pescatarian, gluten_free, dairy_free,
    )

    qvec = embed_query(query)
    d_hits = client.search(
        COLLECTION, query_vector=("text_embedding", qvec),
        query_filter=flt, limit=max(50, limit), with_payload=True
    )

    use_sparse = hybrid and bool(STATE["sparse_enabled"])
    if use_sparse:
        s_hits = client.search(
            COLLECTION, query_vector=("sparse_embedding", to_sparse_vector(query)),
            query_filter=flt, limit=max(50, limit), with_payload=True
        )
        fused = rrf_fuse([d_hits, s_hits])
        top_ids = [pid for pid, _ in fused[:limit]]
        recs = client.retrieve(COLLECTION, ids=top_ids, with_payload=True)
        order = {pid: i for i, (pid, _) in enumerate(fused)}
        recs.sort(key=lambda r: order.get(r.id, 10**9))
        results = [r.payload for r in recs]
    else:
        results = [h.payload for h in d_hits[:limit]]

    return {"hybrid_used": use_sparse, "count": len(results), "results": results}
