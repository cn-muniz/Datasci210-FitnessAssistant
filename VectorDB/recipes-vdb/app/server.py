import os, time, pickle, threading
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel, Field

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Filter, FieldCondition, Range, MatchValue,
    SparseVector, ScoredPoint, NamedSparseVector   # <= add this
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

app = FastAPI(
    title="Recipe Search (Qdrant Cloud)",
    version="0.6.0",
    description="""Minimal, web-ready API for dense (and optional hybrid) recipe search over Qdrant.

**Key endpoints**
- `GET /status` – server readiness
- `POST /search` – dense/hybrid search returning Top-K
- `POST /recipes/top` – convenience endpoint that always returns the single top recipe

**Notes**
- Set environment variables: `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION`, `EMBED_MODEL`, `TFIDF_PATH`
- Exposes a static demo UI at `/`.""",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", tags=["UI"])
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

client: Optional[QdrantClient] = None
embedder: Optional[SentenceTransformer] = None
tfidf: Optional[TfidfVectorizer] = None

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
            sparse_enabled = bool(getattr(cfg, "sparse_vectors", None))
            if not sparse_enabled:
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

# --------------- Pydantic models -----------------

class DietFlags(BaseModel):
    vegan: Optional[bool] = Field(None, description="True to require vegan")
    vegetarian: Optional[bool] = Field(None, description="True to require vegetarian")
    pescatarian: Optional[bool] = Field(None, description="True to require pescatarian")
    gluten_free: Optional[bool] = Field(None, description="True to require gluten-free")
    dairy_free: Optional[bool] = Field(None, description="True to require dairy-free")

class RecipeSearchRequest(BaseModel):
    query: str = Field(..., description="Free-text dense search query")
    limit: int = Field(5, ge=1, le=50, description="How many results to return (Top-K)")

    cal_min: Optional[float] = Field(None, description="Min kcal per serving")
    cal_max: Optional[float] = Field(None, description="Max kcal per serving")

    protein_min: Optional[float] = None
    protein_max: Optional[float] = None

    carbs_min: Optional[float] = None
    carbs_max: Optional[float] = None

    fat_min: Optional[float] = None
    fat_max: Optional[float] = None

    sodium_min: Optional[float] = None
    sodium_max: Optional[float] = None

    cuisine_tag: Optional[str] = Field(None, description="Single cuisine tag to match")
    diet_tag: Optional[str] = Field(None, description="Single diet tag to match")
    meal_tag: Optional[str] = Field(None, description="Meal/course tag to match (e.g., main, breakfast)")

    vegan: Optional[bool] = None
    vegetarian: Optional[bool] = None
    pescatarian: Optional[bool] = None
    gluten_free: Optional[bool] = None
    dairy_free: Optional[bool] = None

    hybrid: bool = Field(False, description="Fuse dense + sparse (if collection has sparse vectors)")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "mediterranean chicken bowl, lemon garlic",
                "limit": 5,
                "cal_min": 450, "cal_max": 700,
                "protein_min": 30,
                "carbs_max": 70,
                "meal_tag": "main",
                "vegetarian": None, "vegan": None,
                "gluten_free": True,
                "hybrid": True
            }
        }

class SearchResponse(BaseModel):
    hybrid_used: bool
    count: int
    results: List[Dict[str, Any]]

# --------------- Helpers -----------------

def embed_query(text: str) -> List[float]:
    if embedder is None:
        raise RuntimeError("Model not loaded yet")
    # e5 expects query prefix; harmless if not used in your embeddings
    v = embedder.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
    return v.astype("float32").tolist()

def _add_range(must_list, key: str, min_v: Optional[float], max_v: Optional[float]):
    if min_v is None and max_v is None:
        return
    must_list.append(FieldCondition(key=key, range=Range(gte=min_v, lte=max_v)))

def make_filter(
    cal_min: Optional[float], cal_max: Optional[float],
    protein_min: Optional[float], protein_max: Optional[float],
    carbs_min: Optional[float], carbs_max: Optional[float],
    fat_min: Optional[float], fat_max: Optional[float],
    sodium_min: Optional[float], sodium_max: Optional[float],
    cuisine_tag: Optional[str], diet_tag: Optional[str], meal_tag: Optional[str],
    vegan: Optional[bool], vegetarian: Optional[bool], pescatarian: Optional[bool],
    gluten_free: Optional[bool], dairy_free: Optional[bool],
) -> Optional[Filter]:
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
    def bmatch(flag_key: str, flag_val: Optional[bool]):
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

def _assert_ready():
    if not (STATE["client_ok"] and STATE["collection_ok"] and STATE["model_loaded"]):
        raise HTTPException(status_code=503, detail={"message": "Server not ready", "status": STATE})

def _core_search(payload: RecipeSearchRequest, force_limit: Optional[int] = None) -> SearchResponse:
    _assert_ready()

    flt = make_filter(
        payload.cal_min, payload.cal_max,
        payload.protein_min, payload.protein_max,
        payload.carbs_min, payload.carbs_max,
        payload.fat_min, payload.fat_max,
        payload.sodium_min, payload.sodium_max,
        payload.cuisine_tag, payload.diet_tag, payload.meal_tag,
        payload.vegan, payload.vegetarian, payload.pescatarian,
        payload.gluten_free, payload.dairy_free,
    )

    qvec = embed_query(payload.query)
    search_limit = max(50, force_limit or payload.limit)

    d_hits = client.search(
        COLLECTION, query_vector=("text_embedding", qvec),
        query_filter=flt, limit=search_limit, with_payload=True
    )

    use_sparse = payload.hybrid and bool(STATE["sparse_enabled"])
    if use_sparse:
        # Build a named sparse vector explicitly (works on older client versions)
        sv = to_sparse_vector(payload.query)
        s_hits = client.search(
            COLLECTION,
            query_vector=NamedSparseVector(name="sparse_embedding", vector=sv),
            query_filter=flt,
            limit=search_limit,
            with_payload=True
        )

        fused = rrf_fuse([d_hits, s_hits])
        top_ids = [pid for pid, _ in fused[: (force_limit or payload.limit)]]
        recs = client.retrieve(COLLECTION, ids=top_ids, with_payload=True)
        order = {pid: i for i, (pid, _) in enumerate(fused)}
        recs.sort(key=lambda r: order.get(r.id, 10**9))
        results = [r.payload for r in recs]
    else:
        results = [h.payload for h in d_hits[: (force_limit or payload.limit)]]

    return SearchResponse(hybrid_used=use_sparse, count=len(results), results=results)

# --------------- Endpoints ----------------

@app.get("/status", tags=["Ops"])
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

@app.post("/search", response_model=SearchResponse, tags=["Search"], summary="Dense (optional hybrid) search for Top-K recipes")
def search(payload: RecipeSearchRequest = Body(...)):
    """Return Top-K results matching the query and filters."""
    return _core_search(payload)

@app.post("/recipes/top", response_model=SearchResponse, tags=["Search"], summary="Return only the single best recipe")
def top_recipe(payload: RecipeSearchRequest = Body(...)):
    """Return exactly one recipe (the top match) for easy web-app calls."""
    return _core_search(payload, force_limit=1)
