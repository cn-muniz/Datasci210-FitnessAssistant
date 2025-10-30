import os, time, pickle, threading, json, requests
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Body, HTTPException, APIRouter
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
import logging
logging.basicConfig(level=logging.INFO)
import llm_planning  # for meal plan prompt templates
import llm_judge

# ---------------- Config ----------------

COLLECTION = os.getenv("QDRANT_COLLECTION", "recipes")

def _get_env_key(env_var: str) -> str:
    raw = os.getenv(env_var, "").strip()
    # handle Secrets Manager key/value JSON or quoted values
    if raw.startswith("{") and ":" in raw:
        try:
            d = json.loads(raw)
            # try common keys
            for k in (env_var, "api_key", "key"):
                if k in d and isinstance(d[k], str):
                    return d[k].strip().strip('"')
        except Exception:
            pass
    return raw.strip().strip('"')

QDRANT_URL = os.getenv("QDRANT_URL", "https://YOUR-CLOUD-URL:6333")
QDRANT_API_KEY = _get_env_key("QDRANT_API_KEY")
logging.info(f"QDRANT_URL: {QDRANT_URL}\nQDRANT_API_KEY: {QDRANT_API_KEY}\n")

COHERE_API_BASE = os.getenv("COHERE_URL", "https://api.cohere.com/v2/chat")
COHERE_API_KEY = _get_env_key("COHERE_API_KEY")
logging.info(f"COHERE_API_BASE: {COHERE_API_BASE}\nCOHERE_API_KEY: {COHERE_API_KEY}\n")

EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/e5-base-v2")
TFIDF_PATH = os.getenv("TFIDF_PATH", "/app_state/tfidf.pkl")

MEAL_PLAN_CONFIG = [
    {"name": "breakfast", "calorie_pct": 0.25, "query": "healthy breakfast", "meal_tag": "breakfast"},
    {"name": "lunch", "calorie_pct": 0.30, "query": "easy lunch salad", "meal_tag": "salad"},
    {"name": "dinner", "calorie_pct": 0.30, "query": "healthy dinner", "meal_tag": "main"},
    {"name": "snack", "calorie_pct": 0.15, "query": "healthy snack", "meal_tag": "snack"},
]
MEAL_PLAN_TOLERANCE = 0.05  # 5.0% wiggle room per course
DIETARY_FLAG_FIELDS = {"vegan", "vegetarian", "pescatarian", "gluten_free", "dairy_free"}

# breakfast, lunch, dinnner
WEEKLY_PLAN_CONFIG = [
    # Day 1
    [{"name": "breakfast", "calorie_pct": 0.3, "query": "healthy pancakes", "meal_tag": "breakfast"},
    {"name": "lunch", "calorie_pct": 0.3, "query": "easy salad", "meal_tag": "salad"},
    {"name": "dinner", "calorie_pct": 0.4, "query": "asian rice meal", "meal_tag": "main"}],
    # Day 2
    [{"name": "breakfast", "calorie_pct": 0.3, "query": "healthy smoothie bowl", "meal_tag": "breakfast"},
    {"name": "lunch", "calorie_pct": 0.3, "query": "easy sandwich", "meal_tag": "main"},
    {"name": "dinner", "calorie_pct": 0.4, "query": "italian pasta meal", "meal_tag": "main"}],
    # Day 3
    [{"name": "breakfast", "calorie_pct": 0.3, "query": "healthy eggs breakfast", "meal_tag": "breakfast"},
    {"name": "lunch", "calorie_pct": 0.3, "query": "easy soup", "meal_tag": "soup"},
    {"name": "dinner", "calorie_pct": 0.4, "query": "mexican rice meal", "meal_tag": "main"}],
    # Day 4
    [{"name": "breakfast", "calorie_pct": 0.3, "query": "healthy oatmeal", "meal_tag": "breakfast"},
    {"name": "lunch", "calorie_pct": 0.3, "query": "easy pasta salad", "meal_tag": "salad"},
    {"name": "dinner", "calorie_pct": 0.4, "query": "american comfort food", "meal_tag": "main"}],
    # Day 5
    [{"name": "breakfast", "calorie_pct": 0.3, "query": "healthy yogurt bowl", "meal_tag": "breakfast"},
    {"name": "lunch", "calorie_pct": 0.3, "query": "easy wrap sandwich", "meal_tag": "main"},
    {"name": "dinner", "calorie_pct": 0.4, "query": "mediterranean meal", "meal_tag": "main"}],
    # Day 6
    [{"name": "breakfast", "calorie_pct": 0.3, "query": "healthy avocado toast", "meal_tag": "breakfast"},
    {"name": "lunch", "calorie_pct": 0.3, "query": "easy grain bowl", "meal_tag": "salad"},
    {"name": "dinner", "calorie_pct": 0.4, "query": "indian curry meal", "meal_tag": "main"}],
]

MEAL_CALORIES_FRAC = {
    "breakfast": 0.3,
    "lunch": 0.3,
    "dinner": 0.4,
}

# --------------- App setup ---------------

app = FastAPI(
    title="Recipe Search (Qdrant Cloud)",
    version="0.6.0",
    description="""Web-ready API for dense (and optional hybrid) recipe search over Qdrant. Also supports multi-day meal planning using a Cohere LLM to generate meal ideas and then searching for recipes to match.

**Key endpoints**
- `GET /status` – server readiness
- `POST /search` – dense/hybrid search returning Top-K
- `POST /recipes/top` – convenience endpoint that always returns the single top recipe
- `POST /mealplan/one-day` – generate a one-day meal plan
- `POST /mealplan/n-day` – generate a multi-day meal plan (1-7 days)
- `POST /test-chat` – test Cohere LLM chat endpoint
- `POST /generate-meal-ideas` – generate a multi-day meal plan using LLM to create meal ideas

**Notes**
- Set environment variables: `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION`, `EMBED_MODEL`, `TFIDF_PATH`, `COHERE_API_KEY`, `COHERE_URL`
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


@app.get("/meal-planner", tags=["UI"])
def meal_planner_ui():
    return FileResponse("static/n_day_planner.html")

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

class MacroSummary(BaseModel):
    protein: str
    carbs: str
    fat: str

class MealSummary(BaseModel):
    title: str
    calories: int
    description: Optional[str] = None
    macros: MacroSummary
    instructions: Optional[str] = None
    ingredients: Optional[List[str]] = None
    quantities: Optional[List[str]] = None
    units: Optional[List[str]] = None
    meal_type: Optional[str] = Field(None, description="Meal slot this recipe fills (breakfast, lunch, etc.)")
    query: Optional[str] = Field(None, description="The search query used to find this recipe")

class MealPlanRequest(BaseModel):
    caloric_target: float = Field(..., gt=0, description="Desired total calories for the day")
    dietary: List[str] = Field(default_factory=list, description="List of dietary flags, e.g. ['gluten_free']")
    preferences: Optional[str] = Field(None, description="Free-text preferences to influence meal selection")
    exclusions: Optional[str] = Field(None, description="Free-text ingredients or tags to avoid")

    class Config:
        json_schema_extra = {
            "example": {
                "caloric_target": 2750,
                "dietary": ["vegetarian", "dairy_free"],
                "preferences": "high protein, low carb",
                "exclusions": "peanuts, mushrooms"
            }
        }

class MealPlanResponse(BaseModel):
    day: Optional[int] = Field(None, description="Day number in multi-day plan, if applicable")
    target_calories: float
    total_calories: float
    target_protein: Optional[float]
    total_protein: Optional[float]
    target_fat: Optional[float]
    total_fat: Optional[float]
    target_carbs: Optional[float]
    total_carbs: Optional[float]
    meals: List[Optional[MealSummary]]

class NDayPlanRequest(BaseModel):
    # caloric target, dietary flags, preferences, exclusions
    target_calories: float = Field(..., gt=0, description="Desired total calories for the day")
    target_protein: Optional[float] = Field(default=100, gt=0, description="Desired total protein for the day")
    target_fat: Optional[float] = Field(default=60, gt=0, description="Desired total fat for the day")
    target_carbs: Optional[float] = Field(default=200, gt=0, description="Desired total carbs for the day")
    dietary: List[str] = Field(default_factory=list, description="List of dietary flags, e.g. ['gluten_free']")
    num_days: int = Field(..., ge=1, le=7, description="Number of days to plan for (1-7)")
    limit_per_meal: int = Field(5, ge=1, le=20, description="How many recipes to fetch per meal slot. One will be chosen randomly.")
    preferences: Optional[str] = Field(None, description="Free-text preferences to influence meal selection")
    exclusions: Optional[str] = Field(None, description="Free-text ingredients or tags to avoid")

    class Config:
        json_schema_extra = {
            "example": {
                "target_calories": 2500,
                "target_protein": 80,
                "target_fat": 50,
                "target_carbs": 250,
                "dietary": ["vegetarian"],
                "num_days": 7,
                "limit_per_meal": 1,
                "preferences": "I want a mix of pancakes and oatmeal for breakfast, some salads for lunch, and hearty dinners with chickpeas or broccoli",
                "exclusions": "peanuts, mushrooms"
            }
        }

class NDayPlanResponse(BaseModel):
    daily_plans: List[MealPlanResponse]

# class MacroTargets(BaseModel):
#     calorie_max_pct: float = Field(..., gt=0, description="Maximum calories for a meal as a percentage of the daily target calories")
#     calorie_min_pct: float = Field(..., gt=0, description="Minimum calories for a meal as a percentage of the daily target calories")
#     protein_max_pct: float = Field(..., gt=0, description="Maximum protein for a meal as a percentage of the daily target protein")
#     protein_min_pct: float = Field(..., gt=0, description="Minimum protein for a meal as a percentage of the daily target protein")
#     fat_max_pct: float = Field(..., gt=0, description="Maximum fat for a meal as a percentage of the daily target fat")
#     fat_min_pct: float = Field(..., gt=0, description="Minimum fat for a meal as a percentage of the daily target fat")
#     carb_max_pct: float = Field(..., gt=0, description="Maximum carb for a meal as a percentage of the daily target carb")
#     carb_min_pct: float = Field(..., gt=0, description="Minimum carb for a meal as a percentage of the daily target carb")

class MealMacroTargets(BaseModel):
    macro_target_pct: float = Field(30, gt=0, description="Target macros for an individual meal as a percentage of the target macros.")
    macro_tolerance_pct: float = Field(5, gt=0, description="Individual meal macro tolerance as a percentage. Works in combination with the macro_target_pct. Final macro range is macro_target*(macro_target_pct +/- macro_tolerance_pct)/100.0")

BREAKFAST_MACRO_DEFAULT = MealMacroTargets(macro_target_pct=30.0, macro_tolerance_pct=5.0)
LUNCH_MACRO_DEFAULT = MealMacroTargets(macro_target_pct=30.0, macro_tolerance_pct=5.0)
DINNER_MACRO_DEFAULT = MealMacroTargets(macro_target_pct=40.0, macro_tolerance_pct=5.0)

class NDayRecipesRequest(BaseModel):
    # caloric target, dietary flags, preferences, exclusions
    target_calories: float = Field(2500, gt=0, description="Desired total calories for the day")
    target_protein: float = Field(100, gt=0, description="Desired total protein for the day")
    target_fat: float = Field(60, gt=0, description="Desired total fat for the day")
    target_carbs: float = Field(200, gt=0, description="Desired total carbs for the day")
    breakfast_targets: MealMacroTargets = Field(BREAKFAST_MACRO_DEFAULT, description="Defines the macro targets for breakfast meals as a percentage of daily target macros")
    lunch_targets: MealMacroTargets = Field(LUNCH_MACRO_DEFAULT, description="Defines the macro targets for lunch meals as a percentage of daily target macros")
    dinner_targets: MealMacroTargets = Field(DINNER_MACRO_DEFAULT, description="Defines the macro targets for dinner meals as a percentage of daily target macros")
    dietary: List[str] = Field(default_factory=list, description="List of dietary flags, e.g. ['gluten_free']. Options are 'vegetarian', 'vegan', 'gluten_free', 'dairy_free', 'pescetarian'")
    num_days: int = Field(..., ge=1, le=7, description="Number of days to plan for (1-7)")
    queries_per_meal: int = Field(2, ge=1, le=14, description="How many recipes to fetch per meal slot. One will be chosen randomly.")
    candidate_recipes: int = Field(50, ge=3, le=100, description="The maximum number of total candidate recipes to be returned")
    preferences: Optional[str] = Field(None, description="Free-text preferences to influence meal selection")
    exclusions: Optional[str] = Field(None, description="Free-text ingredients or tags to avoid")

    class Config:
        json_schema_extra = {
            "example": {
                "target_calories": 2500,
                "target_protein": 100,
                "target_fat": 60,
                "target_carbs": 200,
                "breakfast_targets": {
                    "macro_target_pct": 30,
                    "macro_tolerance_pct": 10
                },
                "lunch_targets": {
                    "macro_target_pct": 30,
                    "macro_tolerance_pct": 10
                },
                "dinner_targets": {
                    "macro_target_pct": 40,
                    "macro_tolerance_pct": 10
                },
                "dietary": ["vegetarian"],
                "num_days": 7,
                "queries_per_meal": 7,
                "candidate_recipes": 42, 
                "preferences": "I want a mix of pancakes and oatmeal for breakfast, some salads for lunch, and hearty dinners with chickpeas or broccoli",
                "exclusions": "peanuts, mushrooms"
            }
        }

class MealForLlm(BaseModel):
    title: str
    calories: float
    protein_g: float
    fat_g: float
    carbs_g: float
    # TODO: Add sodium??

    description: Optional[str] = None
    instructions: Optional[str] = None
    ingredients: Optional[List[str]] = None
    quantities: Optional[List[str]] = None
    units: Optional[List[str]] = None
    meal_type: Optional[str] = Field(None, description="Meal slot this recipe fills (breakfast, lunch, etc.)")
    recipe_id: Optional[str] = Field(None, description="Unique ID from Qdrant vectorDB")
    query: Optional[str] = Field(None, description="Query used in dense search")

class MealCounts(BaseModel):
    breakfast: int = 0
    lunch: int = 0
    dinner: int = 0

class NDayRecipesResponse(BaseModel):
    candidate_recipes: List[MealForLlm] = Field(None, description="A list of candidate recipes returned from the vectorDB")
    original_request: NDayRecipesRequest = Field(None, description="Pass through of the request")
    queries: List[Optional[str]] = Field(None, description="List of dense text queries used in vectorDB search")
    meal_counts: MealCounts = Field(None, description="Number of breakfast, lunch, and dinner candidates returned")

class TestChatRequest(BaseModel):
    messages: Any = Field(..., description="Single string or list of message parts to send to the LLM")
    model: Optional[str] = Field("command-a-03-2025", description="Cohere chat model to use")
    class Config:
        json_schema_extra = {
            "example": {
                "messages": [
                    {"role": "user", "content": "Generate a three brief meal ideas: one breakfast, one lunch, one dinner."}
                ],
                "model": "command-a-03-2025"
            }
        }

class TestChatResponse(BaseModel):
    response: str

class GenerateMealIdeasRequest(BaseModel):
    user_input: str = Field(..., description="User input describing meal preferences")
    num_days: int = Field(..., ge=1, le=14, description="Number of days to plan for (1-14)")
    model: Optional[str] = Field("command-a-03-2025", description="Cohere chat model to use")
    dietary: Optional[list[str]] = Field(None, description="List of dietary flags, e.g. ['gluten_free']. Options are 'vegetarian', 'vegan', 'gluten_free', 'dairy_free', 'pescetarian'")
    class Config:
        json_schema_extra = {
            "example": {
                "user_input": "I want a mix of pancakes and oatmeal for breakfast, some salads for lunch, and hearty dinners with chicken or fish",
                "num_days": 2,
                "model": "command-a-03-2025",
                "dietary": ["vegetarian"]
            }
        }

class GenerateMealIdeasResponse(BaseModel):
    plan_meta: Optional[Dict[str, Any]] = None
    days: Optional[List[Dict[str, Any]]] = None

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
    # logging.info(f"Dense search found {len(d_hits)} hits")

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

    # logging.info(f"Returning {len(results)} results (hybrid={use_sparse})")
    return SearchResponse(hybrid_used=use_sparse, count=len(results), results=results)

def _dietary_kwargs(dietary: List[str]) -> Dict[str, bool]:
    return {flag: True for flag in dietary if flag in DIETARY_FLAG_FIELDS}

def _format_meal_result(recipe_payload: Optional[Dict[str, Any]], meal_type: str, query: str) -> Optional[MealSummary]:
    if not recipe_payload:
        return None

    macros = recipe_payload.get("macros_per_serving", {}) or {}
    def _macro_val(key: str) -> float:
        val = macros.get(key, 0) or 0
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0

    return MealSummary(
        meal_type=meal_type,
        title=recipe_payload.get("title", ""),
        calories=int(_macro_val("cal") + 0.5),
        description=recipe_payload.get("summary"),
        instructions=recipe_payload.get("instructions"),
        ingredients=recipe_payload.get("ingredients_raw"),
        # Sometimes quantities/units are dicitonaries with "text" keys, sometimes they are just lists of strings
        quantities=[q["text"] if isinstance(q, dict) else q for q in recipe_payload.get("quantities")],
        units=[u["text"] if isinstance(u, dict) else u for u in recipe_payload.get("units")],
        query=query,
        macros=MacroSummary(
            protein=f"{int(_macro_val('protein_g') + 0.5)}g",
            carbs=f"{int(_macro_val('carbs_g') + 0.5)}g",
            fat=f"{int(_macro_val('fat_g') + 0.5)}g",
        ),
    )

def _format_meal_candidate(recipe_payload: Optional[Dict[str, Any]], meal_type: str, query: str) -> Optional[MealSummary]:
    if not recipe_payload:
        return None

    return MealSummary(
        meal_type=meal_type,
        title=recipe_payload.get("title", ""),
        calories=int(recipe_payload.get("calories") + 0.5),
        description=recipe_payload.get("description"),
        instructions=recipe_payload.get("instructions"),
        ingredients=recipe_payload.get("ingredients"),
        # Sometimes quantities/units are dicitonaries with "text" keys, sometimes they are just lists of strings
        quantities=[q["text"] if isinstance(q, dict) else q for q in recipe_payload.get("quantities")],
        units=[u["text"] if isinstance(u, dict) else u for u in recipe_payload.get("units")],
        query=query,
        macros=MacroSummary(
            protein=f"{int(recipe_payload.get('protein_g') + 0.5)}g",
            carbs=f"{int(recipe_payload.get('carbs_g') + 0.5)}g",
            fat=f"{int(recipe_payload.get('fat_g') + 0.5)}g",
        ),
    )


def _format_meal_result_llm(recipe_payload: Optional[Dict[str, Any]], meal_type: str, query: str) -> Optional[MealSummary]:
    if not recipe_payload:
        return None

    macros = recipe_payload.get("macros_per_serving", {}) or {}
    def _macro_val(key: str) -> float:
        val = macros.get(key, 0) or 0
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0

    return MealForLlm(
        title=recipe_payload.get("title", ""),
        calories=_macro_val("cal"),
        protein_g=_macro_val("protein_g"),
        fat_g=_macro_val("fat_g"),
        carbs_g=_macro_val("carbs_g"),
        description=recipe_payload.get("summary"),
        instructions=recipe_payload.get("instructions"),
        ingredients=recipe_payload.get("ingredients_raw"),
        quantities=[q["text"] for q in recipe_payload.get("quantities")],
        units=[u["text"] for u in recipe_payload.get("units")],
        meal_type=meal_type,
        recipe_id=recipe_payload.get("source_id"),
        query=query
    )

def _cohere_chat(messages, model="command-a-03-2025"):
    url = COHERE_API_BASE
    # headers = {"Authorization": f"Bearer {COHERE_API_KEY}", "Content-Type": "application/json"}
    headers = {"Authorization": f"Bearer {COHERE_API_KEY}"}
    payload = {
        "model": model,
        "messages": messages,
        "stream": False
    }
    r = requests.post(url, json=payload, headers=headers, timeout=45)
    r.raise_for_status()
    # Cohere v2 chat returns a "message" with "content" parts; join text parts:
    parts = r.json()["message"]["content"]
    return "".join(p.get("text", "") for p in parts)

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

@app.post(
    "/meal-planning/one-day",
    response_model=MealPlanResponse,
    tags=["Meal Planning"],
    summary="Generate a one-day meal plan using caloric target and dietary restrictions",
)
def one_day(payload: MealPlanRequest = Body(...)):
    """Return a simple four-meal day (breakfast, lunch, dinner, snack) tailored to calorie target + diet flags."""
    _assert_ready()

    original_target = float(payload.caloric_target)
    adjusted_target = float(payload.caloric_target)
    dietary_kwargs = _dietary_kwargs(payload.dietary)

    meals: List[Optional[MealSummary]] = []
    total_calories = 0.0

    for cfg in MEAL_PLAN_CONFIG:
        # Add free-text preferences/exclusions to the query
        base_query = cfg["query"]
        if payload.preferences:
            base_query += " preferring: (" + payload.preferences + ")"
        # TODO:  We will eventually handle exclusions with the LLM filtering things out
        if payload.exclusions:
            base_query += " excluding (" + payload.exclusions + ")"

        # logging.info(f"Meal '{cfg['name']}' query: {base_query}")

        pct = cfg["calorie_pct"]
        cal_min = max(0.0, adjusted_target * max(pct - MEAL_PLAN_TOLERANCE, 0))
        cal_max = max(0.0, adjusted_target * (pct + MEAL_PLAN_TOLERANCE))

        request_payload = RecipeSearchRequest(
            query=cfg["query"],
            limit=1,
            meal_tag=cfg["meal_tag"],
            cal_min=cal_min,
            cal_max=cal_max,
            **dietary_kwargs,
        )

        search_response = _core_search(request_payload, force_limit=1)
        recipe_payload = search_response.results[0] if search_response.results else None
        meal_summary = _format_meal_result(recipe_payload, cfg["name"], cfg["query"])
        meals.append(meal_summary)

        actual_calories = 0.0
        if recipe_payload:
            macros = recipe_payload.get("macros_per_serving", {}) or {}
            raw_calories = macros.get("cal", 0) or 0
            try:
                actual_calories = float(raw_calories)
            except (TypeError, ValueError):
                actual_calories = 0.0

        if meal_summary:
            total_calories += meal_summary.calories

        expected_calories = adjusted_target * pct
        calorie_offset = expected_calories - actual_calories
        adjusted_target += calorie_offset

    return MealPlanResponse(
        target_calories=original_target,
        total_calories=total_calories,
        meals=meals,
    )

@app.post(
    "/meal-planning/n-day",
    response_model=NDayPlanResponse,
    tags=["Meal Planning"],
    summary="Generate a n-day meal plan using caloric target and dietary restrictions",
)
def n_day(payload: NDayPlanRequest = Body(...)):
    """Return a simple four-meal day (breakfast, lunch, dinner, snack) tailored to calorie target + diet flags."""
    _assert_ready()

    # original_target = float(payload.target_calories)
    # adjusted_target = float(payload.target_calories)
    # dietary_kwargs = _dietary_kwargs(payload.dietary)

    # First generate the n-day plan using the LLM
    if payload.num_days < 1 or payload.num_days > 7:
        raise HTTPException(status_code=400, detail={"message": "num_days must be between 1 and 7"})
    # generate_meal_ideas_request = GenerateMealIdeasRequest(
    #     user_input=payload.preferences or "balanced meals",
    #     num_days=payload.num_days,
    #     model="command-a-03-2025",
    #     dietary=payload.dietary
    # )
    get_candidate_recipes_request = NDayRecipesRequest(
        target_calories=payload.target_calories,
        target_protein=payload.target_protein,
        target_fat=payload.target_fat,
        target_carbs=payload.target_carbs,
        dietary=payload.dietary,
        num_days=payload.num_days,
        queries_per_meal=7, # queries per meal type - hardcoded for now
        candidate_recipes=42,
        preferences=payload.preferences,
        exclusions=payload.exclusions
    )

    nday_recipes_response = get_candidate_recipes(get_candidate_recipes_request)
    # nday_recipes_response = NDayRecipesResponse(
    #     candidate_recipes=TEST_CANDIDATE_RECIPES_RESPONSE["candidate_recipes"],
    #     original_request=TEST_CANDIDATE_RECIPES_RESPONSE["original_request"],
    #     queries=TEST_CANDIDATE_RECIPES_RESPONSE["queries"],
    #     meal_counts=TEST_CANDIDATE_RECIPES_RESPONSE["meal_counts"]
    # )

    # TODO: consider dropping quantities and units from candidates to free up more tokens
    candidate_recipes_full = nday_recipes_response.candidate_recipes
    # make a copy of candidate_recipes_full with only the fields needed for the LLM judge
    candidate_recipes_trim = [
        {
            "title": recipe.title,
            "calories": recipe.calories,
            "protein_g": recipe.protein_g,
            "fat_g": recipe.fat_g,
            "carbs_g": recipe.carbs_g,
            "description": recipe.description,
            "ingredients": recipe.ingredients,
            "instructions": recipe.instructions,
            "recipe_id": recipe.recipe_id,
            "meal_type": recipe.meal_type
        }
        for recipe in candidate_recipes_full
    ]

    # Create cohere judge prompt
    judge_prompt = llm_judge.generate_system_prompt(
        daily_cal=payload.target_calories,
        daily_protein=payload.target_protein,
        daily_fat=payload.target_fat,
        daily_carbs=payload.target_carbs,
        num_days=payload.num_days,
        dietary=payload.dietary,
        preferences=payload.preferences,
        exclusions=payload.exclusions,
        candidates=json.dumps(candidate_recipes_trim)
    )

    # Call the LLM
    try:
        resp = _cohere_chat(judge_prompt, model="command-a-03-2025")
    except Exception as e:
        raise HTTPException(status_code=500, detail={"message": f"LLM call failed: {type(e).__name__}: {e}"})
    # logging.info(f"LLM response: {resp}")

    selected_recipes = llm_judge.extract_meal_plan_json(resp)
    # # Save selected recipes to a file for debugging
    # with open("selected_recipes.json", "w") as f:
    #     json.dump(selected_recipes, f, indent=2)
    # # Read in the selected recipes from a file for testing
    # with open("selected_recipes.json", "r") as f:
    #     selected_recipes = json.load(f)


    ## Recipe ID mapping
    # Check to see if the recipe_ids are valid
    full_recipe_id_list = [a.recipe_id for a in candidate_recipes_full]

    daily_plans = []

    for day in selected_recipes.get("days"):
        day_meals = {
            "day": day.get("day"),
            "target_calories": payload.target_calories,
            "target_protein": payload.target_protein,
            "target_fat": payload.target_fat,
            "target_carbs": payload.target_carbs,
            "total_calories": 0.0,
            "total_protein": 0.0,
            "total_fat": 0.0,
            "total_carbs": 0.0,
            "meals": []
        }
        for meal_key,meal in day.get("meals").items():
            if meal.get("recipe_id") not in full_recipe_id_list:
                # TODO: Add error handling here
                day_meals.append({
                    "title": "Error",
                    "macros_per_serving": {},
                    "summary": "Error",
                    "instructions": "Error",
                    "ingredients_raw": [],
                    "quantities": [],
                    "units": [],
                    "query": "Error",
                })

            else:
                recipe_index = full_recipe_id_list.index(meal.get("recipe_id"))
                full_meal = candidate_recipes_full[recipe_index]
                # convert to dictionary
                full_meal = full_meal.dict() if isinstance(full_meal, BaseModel) else full_meal
                day_meals.get("meals").append(_format_meal_candidate(full_meal, meal_key, full_meal.get("query", "")))
                # Sum up macros
                day_meals["total_calories"] += full_meal.get("calories", 0) or 0
                day_meals["total_protein"] += full_meal.get("protein_g", 0) or 0
                day_meals["total_fat"] += full_meal.get("fat_g", 0) or 0
                day_meals["total_carbs"] += full_meal.get("carbs_g", 0) or 0

        daily_plans.append(day_meals)
    return NDayPlanResponse(daily_plans=daily_plans)

# LLM test endpoint
@app.post(
    "/test-chat",
    response_model=TestChatResponse,
    tags=["Chat LLM"],
    summary="Test call to Cohere chat LLM",
)
def test_chat(payload: TestChatRequest = Body(...)):
    """Test call to Cohere chat LLM."""
    if not COHERE_API_KEY:
        raise HTTPException(status_code=503, detail={"message": "Cohere API key not configured"})
    try:
        resp = _cohere_chat(payload.messages, model=payload.model or "command-r-plus")
        return TestChatResponse(response=resp)
    except Exception as e:
        raise HTTPException(status_code=500, detail={"message": f"LLM call failed: {type(e).__name__}: {e}"})
    
# Meal ideas generation endpoint
@app.post(
    "/generate-meal-ideas",
    response_model=GenerateMealIdeasResponse,
    tags=["Chat LLM"],
    summary="Generate meal ideas using Cohere chat LLM",
)
def generate_meal_ideas(payload: GenerateMealIdeasRequest = Body(...)):
    """Generate meal ideas using Cohere chat LLM."""
    # We want a nicely formatted list of json objects that each have a query and meal_tag
    if not COHERE_API_KEY:
        raise HTTPException(status_code=503, detail={"message": "Cohere API key not configured"})
    user_input = payload.user_input
    dietary_flags = [flag for flag in (payload.dietary or []) if flag]
    readable_flags = [flag.replace("_", " ").strip() for flag in dietary_flags]
    if readable_flags:
        user_input += (
            " Make sure all queries adhere to the following diet keys: "
            + ", ".join(readable_flags)
            + "."
        )
    # Use llm_planning module for prompt templates
    # prompt = llm_planning.build_messages(payload.num_days, payload.user_input, include_few_shots=False)
    prompt = llm_planning.build_messages(payload.num_days, user_input, include_few_shots=False)
    if readable_flags and prompt and prompt[0].get("role") == "system":
        prompt[0]["content"] = (
            f"{prompt[0]['content']}\n\nDietary constraints to enforce: "
            + ", ".join(readable_flags)
            + "."
        )
    # for item in prompt:
    #     logging.info(item) # Need to keep this short for Cohere
    # logging.info(f"Generated prompt: {prompt}")

    # Call the LLM
    try:
        resp = _cohere_chat(prompt, model=payload.model or "command-a-03-2025")
    except Exception as e:
        raise HTTPException(status_code=500, detail={"message": f"LLM call failed: {type(e).__name__}: {e}"})
    # logging.info(f"LLM response: {resp}")
    # Basic test for now
    # resp = "{\n  \"plan_meta\": {\n    \"days\": 2,\n    \"notes\": \"Mix of pancakes and oatmeal for breakfast, salads for lunch, and hearty chicken or fish dinners.\"\n  },\n  \"days\": [\n    {\n      \"day\": 1,\n      \"breakfast\": {\n        \"title\": \"Blueberry Pancakes\",\n        \"meal_tag\": \"breakfast\",\n        \"query\": \"fluffy blueberry pancakes breakfast\"\n      },\n      \"lunch\": {\n        \"title\": \"Grilled Chicken Caesar Salad\",\n        \"meal_tag\": \"salad\",\n        \"query\": \"grilled chicken caesar salad lunch\"\n      },\n      \"dinner\": {\n        \"title\": \"Baked Salmon with Roasted Vegetables\",\n        \"meal_tag\": \"main\",\n        \"query\": \"baked salmon roasted vegetables hearty dinner\"\n      }\n    },\n    {\n      \"day\": 2,\n      \"breakfast\": {\n        \"title\": \"Apple Cinnamon Oatmeal\",\n        \"meal_tag\": \"breakfast\",\n        \"query\": \"apple cinnamon oatmeal breakfast\"\n      },\n      \"lunch\": {\n        \"title\": \"Greek Salad with Feta\",\n        \"meal_tag\": \"salad\",\n        \"query\": \"greek salad feta lunch\"\n      },\n      \"dinner\": {\n        \"title\": \"Lemon Herb Roasted Chicken\",\n        \"meal_tag\": \"main\",\n        \"query\": \"lemon herb roasted chicken hearty dinner\"\n      }\n    }\n  ]\n}"

    # Validate the LLM response
    valid,format_resp,errors = llm_planning.parse_and_validate_output(resp, payload.num_days)

    if len(errors) > 0:
        logging.warning(f"LLM response parse/validation errors: {errors}")
    if not valid:
        raise HTTPException(status_code=500, detail={"message": f"LLM response format invalid: {format_resp}"})
    # logging.info(f"LLM response parsed OK: {format_resp}")    

    # return the formatted response
    return GenerateMealIdeasResponse(
        plan_meta=format_resp.get("plan_meta"),
        days=format_resp.get("days"),
    )

# Candidate recipe collection endpoint
@app.post(
    "/get-candidate-recipes",
    response_model=NDayRecipesResponse,
    tags=["Chat LLM"],
    summary="Generate unique text dense queries for vectorDB and return candidate recipes",
)
def get_candidate_recipes(payload: NDayRecipesRequest = Body(...)):
    """Generate meal ideas using Cohere chat LLM."""
    # First generate the n-day plan using the LLM
    if payload.num_days < 1 or payload.num_days > 14:
        raise HTTPException(status_code=400, detail={"message": "num_days must be between 1 and 14"})
    
    dietary_kwargs = _dietary_kwargs(payload.dietary)

    generate_meal_ideas_request = GenerateMealIdeasRequest(
        user_input=payload.preferences or "balanced meals",
        num_days = payload.queries_per_meal,
        model="command-a-03-2025", # TODO: Make this an environment variable
        dietary=payload.dietary
    )

    meal_config = generate_meal_ideas(generate_meal_ideas_request)

    # Recipes list
    recipes_list: List[MealForLlm] = []
    queries_list: List[str] = []

    # Determine the number of recipes to return for each meal
    recipes_per_meal = int(np.ceil(payload.candidate_recipes/len(meal_config.days)/3.0))+2 # we will trim back down after collecting recipes

    payload_targets = {
        "breakfast": payload.breakfast_targets,
        "lunch": payload.lunch_targets,
        "dinner": payload.dinner_targets
    }

    # Keep separate lists for each query before collapsing into a single list
    # Nest this under breakfast, lunch, and dinner so we can keep about the same number
    # of candidates for each meal
    nested_candidates = {
        "breakfast": {},
        "lunch": {},
        "dinner": {}
    }

    meal_keys = list(nested_candidates.keys())
    for day_cfg in meal_config.days:
        for meal_key in meal_keys:
            query = day_cfg[meal_key]["query"]
            meal_tag = day_cfg[meal_key]["meal_tag"]
            meal_target = payload_targets[meal_key]

            # Calculate macro targets
            cal_min = payload.target_calories*(meal_target.macro_target_pct-meal_target.macro_tolerance_pct)/100.0
            cal_max = payload.target_calories*(meal_target.macro_target_pct+meal_target.macro_tolerance_pct)/100.0
            protein_min = payload.target_protein*(meal_target.macro_target_pct-meal_target.macro_tolerance_pct)/100.0
            protein_max = payload.target_protein*(meal_target.macro_target_pct+meal_target.macro_tolerance_pct)/100.0
            fat_min = payload.target_fat*(meal_target.macro_target_pct-meal_target.macro_tolerance_pct)/100.0
            fat_max = payload.target_fat*(meal_target.macro_target_pct+meal_target.macro_tolerance_pct)/100.0
            carbs_min = payload.target_carbs*(meal_target.macro_target_pct-meal_target.macro_tolerance_pct)/100.0
            carbs_max = payload.target_carbs*(meal_target.macro_target_pct+meal_target.macro_tolerance_pct)/100.0

            request_payload = RecipeSearchRequest(
                query=query,
                limit=recipes_per_meal,
                meal_tag=meal_tag,
                cal_min=cal_min,
                cal_max=cal_max,
                protein_min=protein_min,
                # protein_max=protein_max,
                # fat_min=fat_min,
                fat_max=fat_max,
                # carbs_min=carbs_min,
                carbs_max=carbs_max,
                **dietary_kwargs,
            )

            search_response = _core_search(request_payload, force_limit=recipes_per_meal)
            # logging.info(f"  Found {search_response.count} results using dense query '{query}'")
            for recipe_payload in search_response.results:
                formatted_recipe = _format_meal_result_llm(recipe_payload,meal_key,query)
                # logging.info(formatted_recipe)
                # recipes_list.append(formatted_recipe)
                recipe_list_key = f"{day_cfg['day']}"
                if recipe_list_key not in list(nested_candidates[meal_key].keys()):
                    nested_candidates[meal_key][recipe_list_key] = []
                nested_candidates[meal_key][recipe_list_key].append(formatted_recipe)

            queries_list.append(query)
    
    # TODO: Add a reranker that takes into account exclusions

    total_candidates = 0
    meal_counts = {
        "breakfast": 0,
        "lunch": 0,
        "dinner": 0
    }
    for meal,query_dict in nested_candidates.items():
        for query,query_list in query_dict.items():
            total_candidates += len(query_list)
            meal_counts[meal] += len(query_list)
    
    # logging.info(f"total_candidates: {total_candidates} desired candidates: {payload.candidate_recipes}")
    while total_candidates > payload.candidate_recipes:
        # logging.info(meal_counts)
        # Get the meal with the most recipes
        max_meal = max(meal_counts,key=meal_counts.get)
        # logging.info(f"max meal: {max_meal}")

        # Get the query with the most recipes for the max meal
        max_query = max(nested_candidates[max_meal],key=lambda k: len(nested_candidates[max_meal][k]))
        # logging.info(f"max query: {max_meal}")

        # Trim the list from the end by 1
        nested_candidates[max_meal][max_query] = nested_candidates[max_meal][max_query][:-1]
        total_candidates -= 1
        meal_counts[max_meal] -= 1

    # Print a summary of the meal counts and collapse into a final candidates list
    meal_counts = {
        "breakfast": 0,
        "lunch": 0,
        "dinner": 0
    }
    for meal,query_dict in nested_candidates.items():
        for query,query_list in query_dict.items():
            meal_counts[meal] += len(query_list)
            for recipe in query_list:
                recipes_list.append(recipe)
    # logging.info(meal_counts)
    # logging.info(f"Total candidate recipes: {len(recipes_list)}")

    return NDayRecipesResponse(
        candidate_recipes=recipes_list,
        original_request=payload,
        queries=queries_list,
        meal_counts=MealCounts(**meal_counts)
    )


# TEST_CANDIDATE_RECIPES_RESPONSE = {
#   "candidate_recipes": [
#     {
#       "title": "Steel Cut Oatmeal and Berries",
#       "calories": 980.1,
#       "protein_g": 77.381,
#       "fat_g": 13.161,
#       "carbs_g": 30.616,
#       "description": "Steel cut oats cooked to desired consistency, topped with yogurt and blueberries; simple, healthy breakfast.",
#       "instructions": "Prepare steel cut oats to package directions and desired consistency.\nPlace 1/4 of cooked oats in breakfast bowl.\nTop with 1/2 cup yogurt.\nTop with 1/4 cup blueberries.\nIf using frozen blueberries, thaw them slightly in the microwave.\nAlso consider blackberries for more soluble fiber!",
#       "ingredients": [
#         "oats",
#         "water, bottled, generic",
#         "blueberries, raw",
#         "yogurt, greek, plain, nonfat"
#       ],
#       "quantities": [
#         "1",
#         "4",
#         "1",
#         "2"
#       ],
#       "units": [
#         "cup",
#         "cup",
#         "cup",
#         "cup"
#       ],
#       "meal_type": "breakfast",
#       "recipe_id": "b9a8cf4750",
#       "query": "vegetarian blueberry oatmeal pancakes breakfast"
#     },
#     {
#       "title": "Vegan Eggnog Waffles",
#       "calories": 720.4480519481,
#       "protein_g": 23.5245454545,
#       "fat_g": 8.4136363636,
#       "carbs_g": 10.0889249639,
#       "description": "Vegan waffles made with oats, eggnog, and maple syrup, crispy on the outside and fluffy on the inside.",
#       "instructions": "Put the oatmeal (either regular or quick cooking) in the food processor and blend until fine.\nAdd the other dry ingredients (flours, spice, and baking powder) and pulse until combined.\nAdd the wet ingredients and pulse until well mixed.\nHeat your waffle maker and grease in whichever way you prefer.\nCook the waffles until crispy.\nI found that it took 5 minutes per batch.\nServe warm with more maple syrup.",
#       "ingredients": [
#         "oats",
#         "wheat flour, white, all-purpose, unenriched",
#         "beans, snap, green, raw",
#         "leavening agents, baking powder, double-acting, sodium aluminum sulfate",
#         "pumpkin, raw",
#         "eggnog",
#         "oil, olive, salad or cooking",
#         "syrup, maple, canadian",
#         "salt, table"
#       ],
#       "quantities": [
#         "2",
#         "12",
#         "12",
#         "1",
#         "12",
#         "2 1/2",
#         "2",
#         "4",
#         "12"
#       ],
#       "units": [
#         "cup",
#         "cup",
#         "cup",
#         "teaspoon",
#         "teaspoon",
#         "cup",
#         "tablespoon",
#         "teaspoon",
#         "teaspoon"
#       ],
#       "meal_type": "breakfast",
#       "recipe_id": "ea4fccb4bf",
#       "query": "vegetarian blueberry oatmeal pancakes breakfast"
#     },
#     {
#       "title": "A Bowl of Gluten-Free Oatmeal",
#       "calories": 721,
#       "protein_g": 34.57,
#       "fat_g": 13.13,
#       "carbs_g": 13.22,
#       "description": "Creamy, comforting oatmeal cooked with milk and water, flavored with vanilla and topped with sweetener and fruit; simple, satisfying breakfast.",
#       "instructions": "Set a saucepan over high heat.\nPour in the milk and water.\nAdd the salt and vanilla extract.\nBring the liquids to a boil.\nWhen the milky water is boiling, pour in the oats.\nStir quite vigorously.\nWhen the water returns to a boil, turn down the heat to low.\nSimmer the oats, stirring every few minutes, until the oats are creamy and plump, the liquid fully absorbed, about 15 minutes.\nTurn off the heat and cover the pan.\nLet the oatmeal sit for five minutes to fully absorb the liquid.\nTop with your favorite sweetener and fruit.\n(This one is maple syrup, peaches, and blackberries.)\nVariations: If you cannot eat dairy, almond milk or hemp milk work well here too.\nIf you have a fresh vanilla bean, scrape the insides of it into the pot instead of vanilla extract.\nThis will be the best oatmeal you have ever eaten.",
#       "ingredients": [
#         "milk, fluid, 1% fat, without added vitamin a and vitamin d",
#         "water, bottled, generic",
#         "salt, table",
#         "vanilla extract",
#         "oats"
#       ],
#       "quantities": [
#         "1",
#         "1",
#         "1/2",
#         "1",
#         "1"
#       ],
#       "units": [
#         "cup",
#         "cup",
#         "teaspoon",
#         "teaspoon",
#         "cup"
#       ],
#       "meal_type": "breakfast",
#       "recipe_id": "a82bdd2944",
#       "query": "vegetarian banana oatmeal breakfast bowl"
#     },
#     {
#       "title": "Oatmeal Banana Muffins - 2 Ww Points",
#       "calories": 705.2103666667,
#       "protein_g": 61.072645,
#       "fat_g": 14.312281,
#       "carbs_g": 56.2251453333,
#       "description": "Moist oatmeal muffins with mashed banana and hint of cinnamon; 2 WW points per serving.",
#       "instructions": "Preheat oven to 400 degrees.\nMix yogurt and oats and set aside.\nAdd dry ingredients (minus sugar) together in another bowl.\nTo the bowl with the yogurt and rolled oats, add egg, applesauce, and then sugar.\nMix welll.\nAdd wet ingredients to the dry ingredients and mix well.\nAdd mashed banana and fold in just to lightly mix.\nBake 20 to 23 minutes.",
#       "ingredients": [
#         "oats",
#         "yogurt, greek, plain, nonfat",
#         "applesauce, canned, unsweetened, without added ascorbic acid (includes usda commodity)",
#         "sugars, brown",
#         "egg substitute, powder",
#         "wheat flour, white, all-purpose, unenriched",
#         "salt, table",
#         "vanilla extract",
#         "spices, cinnamon, ground",
#         "leavening agents, baking soda",
#         "leavening agents, baking powder, double-acting, sodium aluminum sulfate",
#         "bananas, raw"
#       ],
#       "quantities": [
#         "1",
#         "1",
#         "14",
#         "6",
#         "14",
#         "1",
#         "12",
#         "12",
#         "1",
#         "12",
#         "1",
#         "1"
#       ],
#       "units": [
#         "cup",
#         "cup",
#         "cup",
#         "tablespoon",
#         "cup",
#         "cup",
#         "teaspoon",
#         "teaspoon",
#         "pinch",
#         "teaspoon",
#         "teaspoon",
#         "cup"
#       ],
#       "meal_type": "breakfast",
#       "recipe_id": "790eff08ad",
#       "query": "vegetarian banana oatmeal breakfast bowl"
#     },
#     {
#       "title": "Apple Cider Oatmeal",
#       "calories": 828.75,
#       "protein_g": 27.1475,
#       "fat_g": 22.6775,
#       "carbs_g": 22.8025,
#       "description": "Quick, comforting oatmeal cooked in apple cider with cinnamon and butter; easy breakfast.",
#       "instructions": "In a microwave-safe bowl, combine oats and cider.\nMicrowave, uncovered on high for 1-2 minutes.\nStir in the butter, cinnamon and salt (if using.\nI don't like it with salt but hubby does).\nServe.",
#       "ingredients": [
#         "oats",
#         "apples, raw, with skin",
#         "butter, without salt",
#         "spices, cinnamon, ground",
#         "salt, table"
#       ],
#       "quantities": [
#         "1",
#         "1 3/4",
#         "1",
#         "1",
#         "18"
#       ],
#       "units": [
#         "cup",
#         "cup",
#         "tablespoon",
#         "teaspoon",
#         "teaspoon"
#       ],
#       "meal_type": "breakfast",
#       "recipe_id": "30cee67fca",
#       "query": "vegetarian apple cinnamon oatmeal breakfast"
#     },
#     {
#       "title": "Alex's Oatmeal",
#       "calories": 933.2916666667,
#       "protein_g": 35.625,
#       "fat_g": 18.51875,
#       "carbs_g": 23.6279166667,
#       "description": "Baked oatmeal with walnuts, dried apricots and a hint of cinnamon, served with a drizzle of honey.",
#       "instructions": "Preheat oven to 350F (180C).\nSpread walnuts and oats on two rimmed cookie sheets.\nBake for 10 to 12 minutes, until they are golden brown.\nBring water to boil in a saucepan over medium heat.\nStir in oats, apricots, cinnamon, and salt.\nReturn to a boil.\nReduce heat, partially cover, simmer until oats are tender, about 25 minutes, stirring occasionally to prevent sticking.\nMeanwhile, chop the cooled walnuts.\nSpoon oatmeal into 4 bowl, sprinkle with walnuts and drizzle with honey.",
#       "ingredients": [
#         "nuts, walnuts, english",
#         "oats",
#         "water, bottled, generic",
#         "apricots, dried, sulfured, uncooked",
#         "spices, cinnamon, ground",
#         "salt, table",
#         "honey"
#       ],
#       "quantities": [
#         "1/4",
#         "1 1/4",
#         "5",
#         "1",
#         "18",
#         "1/2",
#         "4"
#       ],
#       "units": [
#         "cup",
#         "scoop",
#         "cup",
#         "cup",
#         "teaspoon",
#         "teaspoon",
#         "teaspoon"
#       ],
#       "meal_type": "breakfast",
#       "recipe_id": "1d1a5f004f",
#       "query": "vegetarian apple cinnamon oatmeal breakfast"
#     },
#     {
#       "title": "Eggless Pancakes",
#       "calories": 825.9625,
#       "protein_g": 87.078375,
#       "fat_g": 8.1961875,
#       "carbs_g": 36.658375,
#       "description": "Eggless pancakes made with wheat flour, milk, yogurt, and spices; quick, easy breakfast or brunch option.",
#       "instructions": "Lightly mix all ingredients in a fairly large bowl; leave 5-10 minutes to rise.\nIts important only to mix the batter enough to moisten, as overmixing will make the pancakes tough.\nGently fold down; leave again for 5-10 minutes if you have the time, and gently fold down again.\nCook medium to medium-high on an oiled skillet till golden, carefully breaking apart any large lumps with the spatula.\nServe immediately.\nFor Vegan use only the soy milk and the soy yogurt.",
#       "ingredients": [
#         "wheat flour, white, all-purpose, unenriched",
#         "sugars, granulated",
#         "leavening agents, baking powder, double-acting, sodium aluminum sulfate",
#         "oil, olive, salad or cooking",
#         "salt, table",
#         "milk, fluid, 1% fat, without added vitamin a and vitamin d",
#         "yogurt, greek, plain, nonfat",
#         "spices, cinnamon, ground"
#       ],
#       "quantities": [
#         "2 1/2",
#         "2",
#         "2",
#         "1",
#         "1",
#         "2",
#         "12",
#         "1"
#       ],
#       "units": [
#         "cup",
#         "tablespoon",
#         "tablespoon",
#         "tablespoon",
#         "teaspoon",
#         "cup",
#         "cup",
#         "dash"
#       ],
#       "meal_type": "breakfast",
#       "recipe_id": "70d417df11",
#       "query": "vegetarian chocolate chip pancakes breakfast"
#     },
#     {
#       "title": "Chocolate Myoplex Pancakes",
#       "calories": 740.5296969697,
#       "protein_g": 97.798030303,
#       "fat_g": 18.7790606061,
#       "carbs_g": 27.233,
#       "description": "Protein-rich pancakes made with almond milk, egg substitute, and whey protein powder, cooked on a griddle.",
#       "instructions": "Mix dry ingredients first\nThen add all wet ingredients and mix well.\nHeat a fry pan/griddle and grease LIGHTLY.\nPour small amount of batter (about a 1/4 cup) into pan and spread evenly with a spoon.\nWhen bubbles appear on surface and begin to break, turn over and cook the other side.",
#       "ingredients": [
#         "egg substitute, powder",
#         "beverages, almond milk, unsweetened, shelf stable",
#         "margarine, regular, 80% fat, composite, stick, without salt",
#         "wheat flour, white, all-purpose, unenriched",
#         "sugars, granulated",
#         "leavening agents, baking powder, double-acting, sodium aluminum sulfate",
#         "salt, table",
#         "beverages, protein powder whey based"
#       ],
#       "quantities": [
#         "14",
#         "1",
#         "2",
#         "1",
#         "1",
#         "3",
#         "14",
#         "1"
#       ],
#       "units": [
#         "cup",
#         "cup",
#         "tablespoon",
#         "cup",
#         "tablespoon",
#         "teaspoon",
#         "teaspoon",
#         "scoop"
#       ],
#       "meal_type": "breakfast",
#       "recipe_id": "9666ba3040",
#       "query": "vegetarian chocolate chip pancakes breakfast"
#     },
#     {
#       "title": "A Bowl of Gluten-Free Oatmeal",
#       "calories": 721,
#       "protein_g": 34.57,
#       "fat_g": 13.13,
#       "carbs_g": 13.22,
#       "description": "Creamy, comforting oatmeal cooked with milk and water, flavored with vanilla and topped with sweetener and fruit; simple, satisfying breakfast.",
#       "instructions": "Set a saucepan over high heat.\nPour in the milk and water.\nAdd the salt and vanilla extract.\nBring the liquids to a boil.\nWhen the milky water is boiling, pour in the oats.\nStir quite vigorously.\nWhen the water returns to a boil, turn down the heat to low.\nSimmer the oats, stirring every few minutes, until the oats are creamy and plump, the liquid fully absorbed, about 15 minutes.\nTurn off the heat and cover the pan.\nLet the oatmeal sit for five minutes to fully absorb the liquid.\nTop with your favorite sweetener and fruit.\n(This one is maple syrup, peaches, and blackberries.)\nVariations: If you cannot eat dairy, almond milk or hemp milk work well here too.\nIf you have a fresh vanilla bean, scrape the insides of it into the pot instead of vanilla extract.\nThis will be the best oatmeal you have ever eaten.",
#       "ingredients": [
#         "milk, fluid, 1% fat, without added vitamin a and vitamin d",
#         "water, bottled, generic",
#         "salt, table",
#         "vanilla extract",
#         "oats"
#       ],
#       "quantities": [
#         "1",
#         "1",
#         "1/2",
#         "1",
#         "1"
#       ],
#       "units": [
#         "cup",
#         "cup",
#         "teaspoon",
#         "teaspoon",
#         "cup"
#       ],
#       "meal_type": "breakfast",
#       "recipe_id": "a82bdd2944",
#       "query": "vegetarian strawberry oatmeal smoothie bowl breakfast"
#     },
#     {
#       "title": "Strawberry Freeze Breakfast",
#       "calories": 759.2560400000001,
#       "protein_g": 104.7058398,
#       "fat_g": 5.325006466666667,
#       "carbs_g": 57.0407245,
#       "description": "Frozen strawberry breakfast blend with Greek yogurt and honey; quick, healthy start.",
#       "instructions": "In blender, combine strawberries, milk, yogurt, breakfast drink mix and honey.\nPut cover on blender, and blend until almost smooth, stopping the blender and stirring as necessary.\nPour into 3 or 4 -- 8-ounce glasses.\nServe immediately.",
#       "ingredients": [
#         "strawberries, raw",
#         "milk, fluid, 1% fat, without added vitamin a and vitamin d",
#         "yogurt, greek, plain, nonfat",
#         "vanilla extract",
#         "honey"
#       ],
#       "quantities": [
#         "2",
#         "1 1/2",
#         "12",
#         "2",
#         "2"
#       ],
#       "units": [
#         "cup",
#         "cup",
#         "cup",
#         "ounce",
#         "tablespoon"
#       ],
#       "meal_type": "breakfast",
#       "recipe_id": "8fd3be80c1",
#       "query": "vegetarian strawberry oatmeal smoothie bowl breakfast"
#     },
#     {
#       "title": "Protein pancakes!",
#       "calories": 716.9122887251,
#       "protein_g": 25.6032237401,
#       "fat_g": 13.7429678045,
#       "carbs_g": 29.932542045,
#       "description": "Protein-enriched pancakes made with whey protein powder, peanut butter and milk, cooked on a hot pan; quick, protein-packed breakfast.",
#       "instructions": "Add all ingredients to a bowl mix well.\nPour one cup on hot pan (med-high) flip, garnish, and serve.",
#       "ingredients": [
#         "pancakes, plain, dry mix, complete (includes buttermilk)",
#         "beverages, protein powder whey based",
#         "peanut butter, smooth style, without salt",
#         "vanilla extract",
#         "spices, cinnamon, ground",
#         "milk, fluid, 1% fat, without added vitamin a and vitamin d",
#         "water, bottled, generic"
#       ],
#       "quantities": [
#         "2",
#         "2",
#         "2",
#         "1",
#         "1",
#         "1/2",
#         "1 1/3"
#       ],
#       "units": [
#         "cup",
#         "tablespoon",
#         "tablespoon",
#         "dash",
#         "pinch",
#         "cup",
#         "cup"
#       ],
#       "meal_type": "breakfast",
#       "recipe_id": "b8a96ef9d3",
#       "query": "vegetarian peanut butter banana pancakes breakfast"
#     },
#     {
#       "title": "Peanut Butter Biscuits",
#       "calories": 960.5416666667,
#       "protein_g": 43.73875,
#       "fat_g": 21.79875,
#       "carbs_g": 39.1375,
#       "description": "Flourless peanut butter biscuits made with oat and soy flours, perfect for a gluten-free breakfast or snack.",
#       "instructions": "Preheat oven to 400.\nIn a mixing bowl, combine oat flour, soy flour and baking powder.\nIn a blender, blend peanut butter and milk.\nPour peanut butter mixture into dry ingredients and mix well.\nTurn dough out onto a lightly floured surface and knead lightly.\nRoll out dough 1/4 inches.\nthick and cut into squares or use a cookie cutter.\nPlace biscuits on baking sheet about 1/2 inches.\napart and bake for 15 minutes, or until lightly browned.\nBiscuits should be refrigerated or frozen.",
#       "ingredients": [
#         "wheat flour, white, all-purpose, unenriched",
#         "wheat flour, white, all-purpose, unenriched",
#         "leavening agents, baking powder, double-acting, sodium aluminum sulfate",
#         "peanut butter, smooth style, without salt",
#         "milk, fluid, 1% fat, without added vitamin a and vitamin d"
#       ],
#       "quantities": [
#         "1 1/2",
#         "12",
#         "1",
#         "1 1/4",
#         "34"
#       ],
#       "units": [
#         "cup",
#         "cup",
#         "tablespoon",
#         "cup",
#         "cup"
#       ],
#       "meal_type": "breakfast",
#       "recipe_id": "6465cda3e3",
#       "query": "vegetarian peanut butter banana pancakes breakfast"
#     },
#     {
#       "title": "Diabetic Maple and Ginger Oatmeal",
#       "calories": 822.1428571429,
#       "protein_g": 29.23,
#       "fat_g": 12.2,
#       "carbs_g": 24.8597619048,
#       "description": "Simple oatmeal with ground ginger and maple syrup, suitable for diabetic diets.",
#       "instructions": "Mix the ingredients together.\nSweeten with Splenda if needed.",
#       "ingredients": [
#         "oats",
#         "spices, ginger, ground",
#         "syrup, maple, canadian"
#       ],
#       "quantities": [
#         "1",
#         "18",
#         "2"
#       ],
#       "units": [
#         "cup",
#         "teaspoon",
#         "tablespoon"
#       ],
#       "meal_type": "breakfast",
#       "recipe_id": "91b1cb26c6",
#       "query": "vegetarian maple walnut oatmeal breakfast"
#     },
#     {
#       "title": "Alex's Oatmeal",
#       "calories": 933.2916666667,
#       "protein_g": 35.625,
#       "fat_g": 18.51875,
#       "carbs_g": 23.6279166667,
#       "description": "Baked oatmeal with walnuts, dried apricots and a hint of cinnamon, served with a drizzle of honey.",
#       "instructions": "Preheat oven to 350F (180C).\nSpread walnuts and oats on two rimmed cookie sheets.\nBake for 10 to 12 minutes, until they are golden brown.\nBring water to boil in a saucepan over medium heat.\nStir in oats, apricots, cinnamon, and salt.\nReturn to a boil.\nReduce heat, partially cover, simmer until oats are tender, about 25 minutes, stirring occasionally to prevent sticking.\nMeanwhile, chop the cooled walnuts.\nSpoon oatmeal into 4 bowl, sprinkle with walnuts and drizzle with honey.",
#       "ingredients": [
#         "nuts, walnuts, english",
#         "oats",
#         "water, bottled, generic",
#         "apricots, dried, sulfured, uncooked",
#         "spices, cinnamon, ground",
#         "salt, table",
#         "honey"
#       ],
#       "quantities": [
#         "1/4",
#         "1 1/4",
#         "5",
#         "1",
#         "18",
#         "1/2",
#         "4"
#       ],
#       "units": [
#         "cup",
#         "scoop",
#         "cup",
#         "cup",
#         "teaspoon",
#         "teaspoon",
#         "teaspoon"
#       ],
#       "meal_type": "breakfast",
#       "recipe_id": "1d1a5f004f",
#       "query": "vegetarian maple walnut oatmeal breakfast"
#     },
#     {
#       "title": "Curried Lentil Salad",
#       "calories": 831.1375,
#       "protein_g": 59.578875,
#       "fat_g": 3.528875,
#       "carbs_g": 11.7295,
#       "description": "Mixed salad of lentils, corn, peas and yogurt in a warm, slightly spicy curry sauce; healthy, easy side or light lunch.",
#       "instructions": "Place everything together and mix.",
#       "ingredients": [
#         "lentils, raw",
#         "yogurt, greek, plain, nonfat",
#         "spices, curry powder",
#         "corn, sweet, white, raw",
#         "peas, green, frozen, unprepared"
#       ],
#       "quantities": [
#         "4",
#         "1",
#         "1",
#         "2",
#         "2"
#       ],
#       "units": [
#         "cup",
#         "cup",
#         "teaspoon",
#         "cup",
#         "cup"
#       ],
#       "meal_type": "lunch",
#       "recipe_id": "31c2c820fd",
#       "query": "vegetarian greek salad feta lunch"
#     },
#     {
#       "title": "Chickpea, Pesto & Red Onion Salad",
#       "calories": 672.325,
#       "protein_g": 24.251125,
#       "fat_g": 11.092875,
#       "carbs_g": 71.251375,
#       "description": "Chickpeas, red onions, and pesto come together in a simple, herby salad; best served at room temperature.",
#       "instructions": "Drain and wash chick peas.\nIn a bowl, whisk pesto, olive oil, and lemon juice together.\nAdd chick peas and red onions and toss together.\nBest served at room temperature.",
#       "ingredients": [
#         "chickpeas (garbanzo beans, bengal gram), mature seeds, raw",
#         "sauce, pesto, ready-to-serve, refrigerated",
#         "oil, olive, salad or cooking",
#         "lemon juice, raw",
#         "fresh red onions, upc: 888670013229"
#       ],
#       "quantities": [
#         "2",
#         "3",
#         "1",
#         "1",
#         "34"
#       ],
#       "units": [
#         "cup",
#         "tablespoon",
#         "tablespoon",
#         "tablespoon",
#         "cup"
#       ],
#       "meal_type": "lunch",
#       "recipe_id": "c4e1ad006c",
#       "query": "vegetarian greek salad feta lunch"
#     },
#     {
#       "title": "Quinoa Salad",
#       "calories": 667.225,
#       "protein_g": 26.7655,
#       "fat_g": 8.56175,
#       "carbs_g": 28.4585,
#       "description": "Cooked quinoa mixed with a citrus-herb dressing and sweet corn, onions, and peppers; light, refreshing salad.",
#       "instructions": "Bring water to a boil.\nCook quinoa.\nIn a bowl, combine all the other ingredients.\nWhen quinoa has cooled fold into dressing and vegetables.\nServe at room temperature or cold.",
#       "ingredients": [
#         "quinoa, uncooked",
#         "water, bottled, generic",
#         "orange juice, raw",
#         "lime juice, raw",
#         "vinegar, balsamic",
#         "salt, table",
#         "corn, sweet, white, raw",
#         "onions, spring or scallions (includes tops and bulb), raw",
#         "peppers, sweet, green, raw",
#         "fresh red onions, upc: 888670013229"
#       ],
#       "quantities": [
#         "1 1/2",
#         "2 1/2",
#         "6",
#         "2",
#         "1",
#         "14",
#         "12",
#         "12",
#         "12",
#         "2"
#       ],
#       "units": [
#         "cup",
#         "cup",
#         "tablespoon",
#         "tablespoon",
#         "tablespoon",
#         "teaspoon",
#         "cup",
#         "cup",
#         "cup",
#         "tablespoon"
#       ],
#       "meal_type": "lunch",
#       "recipe_id": "86a1200554",
#       "query": "vegetarian quinoa avocado salad lunch"
#     },
#     {
#       "title": "Chickpea, Pesto & Red Onion Salad",
#       "calories": 672.325,
#       "protein_g": 24.251125,
#       "fat_g": 11.092875,
#       "carbs_g": 71.251375,
#       "description": "Chickpeas, red onions, and pesto come together in a simple, herby salad; best served at room temperature.",
#       "instructions": "Drain and wash chick peas.\nIn a bowl, whisk pesto, olive oil, and lemon juice together.\nAdd chick peas and red onions and toss together.\nBest served at room temperature.",
#       "ingredients": [
#         "chickpeas (garbanzo beans, bengal gram), mature seeds, raw",
#         "sauce, pesto, ready-to-serve, refrigerated",
#         "oil, olive, salad or cooking",
#         "lemon juice, raw",
#         "fresh red onions, upc: 888670013229"
#       ],
#       "quantities": [
#         "2",
#         "3",
#         "1",
#         "1",
#         "34"
#       ],
#       "units": [
#         "cup",
#         "tablespoon",
#         "tablespoon",
#         "tablespoon",
#         "cup"
#       ],
#       "meal_type": "lunch",
#       "recipe_id": "c4e1ad006c",
#       "query": "vegetarian quinoa avocado salad lunch"
#     },
#     {
#       "title": "Salad Supreme Seasoning",
#       "calories": 512.7187179668,
#       "protein_g": 24.4715595529,
#       "fat_g": 22.5967166198,
#       "carbs_g": 2.9790624359,
#       "description": "Customizable salad seasoning blend with sesame seeds, paprika, garlic powder, and parmesan.",
#       "instructions": "Combine all ingredients and mix well.\nStore in a sealable container in the fridge.",
#       "ingredients": [
#         "seeds, sesame seeds, whole, dried",
#         "spices, paprika",
#         "salt, table",
#         "spices, poppy seed",
#         "celery, raw",
#         "spices, garlic powder",
#         "spices, pepper, black",
#         "spices, pepper, red or cayenne",
#         "cheese, parmesan, hard"
#       ],
#       "quantities": [
#         "1 1/2",
#         "1",
#         "34",
#         "12",
#         "12",
#         "14",
#         "14",
#         "1",
#         "2"
#       ],
#       "units": [
#         "teaspoon",
#         "teaspoon",
#         "teaspoon",
#         "teaspoon",
#         "teaspoon",
#         "teaspoon",
#         "teaspoon",
#         "dash",
#         "tablespoon"
#       ],
#       "meal_type": "lunch",
#       "recipe_id": "f92cc55841",
#       "query": "vegetarian caprese salad balsamic glaze lunch"
#     },
#     {
#       "title": "Chickpea, Pesto & Red Onion Salad",
#       "calories": 672.325,
#       "protein_g": 24.251125,
#       "fat_g": 11.092875,
#       "carbs_g": 71.251375,
#       "description": "Chickpeas, red onions, and pesto come together in a simple, herby salad; best served at room temperature.",
#       "instructions": "Drain and wash chick peas.\nIn a bowl, whisk pesto, olive oil, and lemon juice together.\nAdd chick peas and red onions and toss together.\nBest served at room temperature.",
#       "ingredients": [
#         "chickpeas (garbanzo beans, bengal gram), mature seeds, raw",
#         "sauce, pesto, ready-to-serve, refrigerated",
#         "oil, olive, salad or cooking",
#         "lemon juice, raw",
#         "fresh red onions, upc: 888670013229"
#       ],
#       "quantities": [
#         "2",
#         "3",
#         "1",
#         "1",
#         "34"
#       ],
#       "units": [
#         "cup",
#         "tablespoon",
#         "tablespoon",
#         "tablespoon",
#         "cup"
#       ],
#       "meal_type": "lunch",
#       "recipe_id": "c4e1ad006c",
#       "query": "vegetarian caprese salad balsamic glaze lunch"
#     },
#     {
#       "title": "Quinoa Salad",
#       "calories": 667.225,
#       "protein_g": 26.7655,
#       "fat_g": 8.56175,
#       "carbs_g": 28.4585,
#       "description": "Cooked quinoa mixed with a citrus-herb dressing and sweet corn, onions, and peppers; light, refreshing salad.",
#       "instructions": "Bring water to a boil.\nCook quinoa.\nIn a bowl, combine all the other ingredients.\nWhen quinoa has cooled fold into dressing and vegetables.\nServe at room temperature or cold.",
#       "ingredients": [
#         "quinoa, uncooked",
#         "water, bottled, generic",
#         "orange juice, raw",
#         "lime juice, raw",
#         "vinegar, balsamic",
#         "salt, table",
#         "corn, sweet, white, raw",
#         "onions, spring or scallions (includes tops and bulb), raw",
#         "peppers, sweet, green, raw",
#         "fresh red onions, upc: 888670013229"
#       ],
#       "quantities": [
#         "1 1/2",
#         "2 1/2",
#         "6",
#         "2",
#         "1",
#         "14",
#         "12",
#         "12",
#         "12",
#         "2"
#       ],
#       "units": [
#         "cup",
#         "cup",
#         "tablespoon",
#         "tablespoon",
#         "tablespoon",
#         "teaspoon",
#         "cup",
#         "cup",
#         "cup",
#         "tablespoon"
#       ],
#       "meal_type": "lunch",
#       "recipe_id": "86a1200554",
#       "query": "vegetarian kale quinoa salad lunch"
#     },
#     {
#       "title": "Curried Lentil Salad",
#       "calories": 831.1375,
#       "protein_g": 59.578875,
#       "fat_g": 3.528875,
#       "carbs_g": 11.7295,
#       "description": "Mixed salad of lentils, corn, peas and yogurt in a warm, slightly spicy curry sauce; healthy, easy side or light lunch.",
#       "instructions": "Place everything together and mix.",
#       "ingredients": [
#         "lentils, raw",
#         "yogurt, greek, plain, nonfat",
#         "spices, curry powder",
#         "corn, sweet, white, raw",
#         "peas, green, frozen, unprepared"
#       ],
#       "quantities": [
#         "4",
#         "1",
#         "1",
#         "2",
#         "2"
#       ],
#       "units": [
#         "cup",
#         "cup",
#         "teaspoon",
#         "cup",
#         "cup"
#       ],
#       "meal_type": "lunch",
#       "recipe_id": "31c2c820fd",
#       "query": "vegetarian kale quinoa salad lunch"
#     },
#     {
#       "title": "Chickpea, Pesto & Red Onion Salad",
#       "calories": 672.325,
#       "protein_g": 24.251125,
#       "fat_g": 11.092875,
#       "carbs_g": 71.251375,
#       "description": "Chickpeas, red onions, and pesto come together in a simple, herby salad; best served at room temperature.",
#       "instructions": "Drain and wash chick peas.\nIn a bowl, whisk pesto, olive oil, and lemon juice together.\nAdd chick peas and red onions and toss together.\nBest served at room temperature.",
#       "ingredients": [
#         "chickpeas (garbanzo beans, bengal gram), mature seeds, raw",
#         "sauce, pesto, ready-to-serve, refrigerated",
#         "oil, olive, salad or cooking",
#         "lemon juice, raw",
#         "fresh red onions, upc: 888670013229"
#       ],
#       "quantities": [
#         "2",
#         "3",
#         "1",
#         "1",
#         "34"
#       ],
#       "units": [
#         "cup",
#         "tablespoon",
#         "tablespoon",
#         "tablespoon",
#         "cup"
#       ],
#       "meal_type": "lunch",
#       "recipe_id": "c4e1ad006c",
#       "query": "vegetarian mediterranean chickpea salad lunch"
#     },
#     {
#       "title": "Curried Lentil Salad",
#       "calories": 831.1375,
#       "protein_g": 59.578875,
#       "fat_g": 3.528875,
#       "carbs_g": 11.7295,
#       "description": "Mixed salad of lentils, corn, peas and yogurt in a warm, slightly spicy curry sauce; healthy, easy side or light lunch.",
#       "instructions": "Place everything together and mix.",
#       "ingredients": [
#         "lentils, raw",
#         "yogurt, greek, plain, nonfat",
#         "spices, curry powder",
#         "corn, sweet, white, raw",
#         "peas, green, frozen, unprepared"
#       ],
#       "quantities": [
#         "4",
#         "1",
#         "1",
#         "2",
#         "2"
#       ],
#       "units": [
#         "cup",
#         "cup",
#         "teaspoon",
#         "cup",
#         "cup"
#       ],
#       "meal_type": "lunch",
#       "recipe_id": "31c2c820fd",
#       "query": "vegetarian mediterranean chickpea salad lunch"
#     },
#     {
#       "title": "Fresh Corn and Tomato Salad",
#       "calories": 528.391975,
#       "protein_g": 25.1718828333,
#       "fat_g": 11.89679,
#       "carbs_g": 29.7021348333,
#       "description": "Cool corn and cherry tomatoes mixed with lemon vinaigrette, red onion, and basil; refreshing summer salad.",
#       "instructions": "Prepare an ice water bath by filling a large bowl halfway with ice and water; set aside.\nBring a large pot of heavily salted water to a boil over high heat.\nAdd the corn kernels and cook until tender, about 4 minutes.\nDrain and place in the ice water bath until cool, about 4 minutes.\nCombine the lemon juice, salt, and pepper in a large, nonreactive bowl.\nWhile continuously whisking, add the oil in a steady stream until completely incorporated.\nAdd the remaining ingredients and the cooled corn and toss until well coated.\nServe.",
#       "ingredients": [
#         "corn, sweet, white, raw",
#         "lemon juice, raw",
#         "salt, table",
#         "spices, pepper, black",
#         "oil, olive, salad or cooking",
#         "cherries, sweet, raw",
#         "fresh red onions, upc: 888670013229",
#         "spices, basil, dried"
#       ],
#       "quantities": [
#         "5",
#         "2",
#         "1 1/2",
#         "14",
#         "3",
#         "10",
#         "12",
#         "14"
#       ],
#       "units": [
#         "cup",
#         "tablespoon",
#         "teaspoon",
#         "teaspoon",
#         "tablespoon",
#         "ounce",
#         "cup",
#         "cup"
#       ],
#       "meal_type": "lunch",
#       "recipe_id": "be36d4b236",
#       "query": "vegetarian spinach strawberry salad lunch"
#     },
#     {
#       "title": "Crunchy Broccoli Salad",
#       "calories": 690.9382352941,
#       "protein_g": 85.5512941176,
#       "fat_g": 5.5954117647,
#       "carbs_g": 55.2271764706,
#       "description": "Raw broccoli, carrots, celery, sunflower seeds, raisins, and cranberries in a creamy yogurt dressing; crunchy, healthy salad.",
#       "instructions": "Chop broccoli into 1/2-3/4\" pieces.\nFinely dice Celery & Red Onion.\nShred Carrot.\nCombine all dry ingredients in bowl.\nWhisk Yogurt/OJ/Mayo in small bowl pour over dry mixture, toss to coat.",
#       "ingredients": [
#         "broccoli, raw",
#         "fresh red onions, upc: 888670013229",
#         "carrots, raw",
#         "celery, raw",
#         "seeds, sunflower seed kernels, dried",
#         "raisins, seeded",
#         "cranberries, dried, sweetened",
#         "yogurt, greek, plain, nonfat",
#         "yogurt, greek, plain, nonfat",
#         "orange juice, raw",
#         "salad dressing, mayonnaise, regular"
#       ],
#       "quantities": [
#         "4",
#         "14",
#         "14",
#         "14",
#         "2",
#         "2",
#         "2",
#         "13",
#         "14",
#         "2",
#         "1"
#       ],
#       "units": [
#         "cup",
#         "cup",
#         "cup",
#         "cup",
#         "tablespoon",
#         "tablespoon",
#         "tablespoon",
#         "cup",
#         "cup",
#         "tablespoon",
#         "tablespoon"
#       ],
#       "meal_type": "lunch",
#       "recipe_id": "88eba8e160",
#       "query": "vegetarian spinach strawberry salad lunch"
#     },
#     {
#       "title": "Fresh Corn and Tomato Salad",
#       "calories": 528.391975,
#       "protein_g": 25.1718828333,
#       "fat_g": 11.89679,
#       "carbs_g": 29.7021348333,
#       "description": "Cool corn and cherry tomatoes mixed with lemon vinaigrette, red onion, and basil; refreshing summer salad.",
#       "instructions": "Prepare an ice water bath by filling a large bowl halfway with ice and water; set aside.\nBring a large pot of heavily salted water to a boil over high heat.\nAdd the corn kernels and cook until tender, about 4 minutes.\nDrain and place in the ice water bath until cool, about 4 minutes.\nCombine the lemon juice, salt, and pepper in a large, nonreactive bowl.\nWhile continuously whisking, add the oil in a steady stream until completely incorporated.\nAdd the remaining ingredients and the cooled corn and toss until well coated.\nServe.",
#       "ingredients": [
#         "corn, sweet, white, raw",
#         "lemon juice, raw",
#         "salt, table",
#         "spices, pepper, black",
#         "oil, olive, salad or cooking",
#         "cherries, sweet, raw",
#         "fresh red onions, upc: 888670013229",
#         "spices, basil, dried"
#       ],
#       "quantities": [
#         "5",
#         "2",
#         "1 1/2",
#         "14",
#         "3",
#         "10",
#         "12",
#         "14"
#       ],
#       "units": [
#         "cup",
#         "tablespoon",
#         "teaspoon",
#         "teaspoon",
#         "tablespoon",
#         "ounce",
#         "cup",
#         "cup"
#       ],
#       "meal_type": "lunch",
#       "recipe_id": "be36d4b236",
#       "query": "vegetarian tomato mozzarella salad lunch"
#     },
#     {
#       "title": "Chickpea, Pesto & Red Onion Salad",
#       "calories": 672.325,
#       "protein_g": 24.251125,
#       "fat_g": 11.092875,
#       "carbs_g": 71.251375,
#       "description": "Chickpeas, red onions, and pesto come together in a simple, herby salad; best served at room temperature.",
#       "instructions": "Drain and wash chick peas.\nIn a bowl, whisk pesto, olive oil, and lemon juice together.\nAdd chick peas and red onions and toss together.\nBest served at room temperature.",
#       "ingredients": [
#         "chickpeas (garbanzo beans, bengal gram), mature seeds, raw",
#         "sauce, pesto, ready-to-serve, refrigerated",
#         "oil, olive, salad or cooking",
#         "lemon juice, raw",
#         "fresh red onions, upc: 888670013229"
#       ],
#       "quantities": [
#         "2",
#         "3",
#         "1",
#         "1",
#         "34"
#       ],
#       "units": [
#         "cup",
#         "tablespoon",
#         "tablespoon",
#         "tablespoon",
#         "cup"
#       ],
#       "meal_type": "lunch",
#       "recipe_id": "c4e1ad006c",
#       "query": "vegetarian tomato mozzarella salad lunch"
#     },
#     {
#       "title": "Vegan Soy, Lentil, & Veggie Burger Sliders Using Tvp",
#       "calories": 889.609375,
#       "protein_g": 46.81171875,
#       "fat_g": 21.841796875,
#       "carbs_g": 15.588984375,
#       "description": "Vegan burgers made with TVP, lentils, and vegetables, bound with flaxseed and baked into crispy patties; versatile and customizable.",
#       "instructions": "If you're using dry lentils on the stovetop and not the pressure cooker, bring 1/2 cup water to a boil, add the lentils, and lower the heat to medium and let simmer for 20 minutes; drain any excess water when done cooking.\nNuke the vegetable broth for about 2 minutes depending on your microwave's strength.\nYou want it to be near-boiling but not too hot to handle.\nAdd all the dry spices listed with the nutritional yeast under \"Base For The Sliders\" to the heated vegetable broth.\nPour into a large mixing bowl.\nAdd the TVP, making sure it's all covered.\nLet it sit for 10 minutes in order to expand, hydrate, and soak up all the seasonings.\nWhile you wait for that, finely shred the carrot if not using pre-shredded.\nIf not using pre-chopped broccoli like those frozen packages, finely chop up about 1/3 head of a broccoli to get about 1/2 cup of finely chopped broccoli pieces.\nPut the broccoli aside for now.\nHeat up the olive oil in a skillet and sautee the onion, garlic, carrot, and broccoli together until nicely browned, about 5 minutes.\nThe TVP should be all soaked up by now.\nStir it to make sure all the seasoning dispersed.\nAdd the lentils and sauteed veggies.\nMix thoroughly.\nBind with the flax seed.\nFeel free to add some bread crumbs for extra crunch, or more flax if it's not binding well enough.\nThis part was tricky for me because the original mix didn't bind at all.\nI don't recommend using eggs or egg whites to bind because it'll make the mix even more liquidy.\nPreheat the oven to 425F Put a little olive oil on a cookie sheet-- cooking spray just won't work the same for this-- about a teaspoon or a little less, and brush it on evenly with a BBQ/pastry brush or paper towel.\nWhile the oven preheats, let the mixture sit out for about 10 minutes.\nEtract the mixture and put it on the cookie sheet.\nCarefully squish it out evenly, sort of like you're stretching out pizza dough.\nIt should be a large thin rectangle.\nBake for 10-12 minutes or until golden brown on top.\nFlip it over so the other side can cook for another 10-12 minutes.\nUse a pizza cutter to cut into slider squares, you'll get about 16.\nServe on wheat rolls with lettuce, tomato, Vegenaise, Soy Kaas, avocado slices, and/or your other favorite vegetarian \"burger\" fixings!\nIf you want to make these as normal veggie burgers, you'll get 4 big patties and have to bake for about 15 minutes on each side.",
#       "ingredients": [
#         "shortening, vegetable, household, composite",
#         "soup, vegetable broth, ready to serve",
#         "leavening agents, yeast, baker's, active dry",
#         "spices, paprika",
#         "salt, table",
#         "spices, marjoram, dried",
#         "spices, onion powder",
#         "spices, pepper, black",
#         "mustard, prepared, yellow",
#         "spices, pepper, red or cayenne",
#         "onions, raw",
#         "spices, garlic powder",
#         "oil, olive, salad or cooking",
#         "broccoli, raw",
#         "carrots, raw",
#         "lentils, raw",
#         "seeds, flaxseed"
#       ],
#       "quantities": [
#         "1",
#         "1",
#         "2",
#         "14",
#         "14",
#         "14",
#         "18",
#         "18",
#         "18",
#         "1",
#         "14",
#         "12",
#         "2",
#         "12",
#         "12",
#         "13",
#         "1"
#       ],
#       "units": [
#         "cup",
#         "cup",
#         "tablespoon",
#         "teaspoon",
#         "teaspoon",
#         "teaspoon",
#         "teaspoon",
#         "teaspoon",
#         "teaspoon",
#         "dash",
#         "cup",
#         "teaspoon",
#         "tablespoon",
#         "cup",
#         "cup",
#         "cup",
#         "cup"
#       ],
#       "meal_type": "dinner",
#       "recipe_id": "266444cabe",
#       "query": "vegetarian chickpea spinach curry dinner"
#     },
#     {
#       "title": "Ppk's Simple Italian Sausages- Vegan/Vegetarian",
#       "calories": 1244.34375,
#       "protein_g": 154.10875,
#       "fat_g": 28.26265625,
#       "carbs_g": 5.48046875,
#       "description": "Vegan Italian sausages made with vital wheat gluten, beans, and spices, steamed to perfection; easy, plant-based alternative to traditional sausages.",
#       "instructions": "Get your tinfoil squares ready.\nIn a large bowl, mash the beans until no whole ones are left.\nThrow all the other ingredients together in the order listed and mix with a fork (I find it easier to use my hands).\nGet your steaming apparatus ready.\nDivide dough into 8 even parts and form into little logs (about the size of a large hot dog.\n).\nTightly wrap the dough in tin foil, like a tootsie roll, twisting the ends.\n(I didn't wrap mine tight enough last time and they came out strangely shaped but they still cooked through and tasted great.\n).\nPlace wrapped sausages in steamer (I use one of those cheap steamer baskets you can buy at the grocery store) and steam for 40 minutes.\nUnwrap and enjoy or refrigerate for later (they also freeze well).",
#       "ingredients": [
#         "beans, snap, green, raw",
#         "soup, vegetable broth, ready to serve",
#         "oil, olive, salad or cooking",
#         "soy sauce made from soy (tamari)",
#         "vital wheat gluten",
#         "leavening agents, yeast, baker's, active dry",
#         "spices, garlic powder",
#         "spices, fennel seed",
#         "spices, pepper, red or cayenne",
#         "spices, paprika",
#         "spices, oregano, dried",
#         "spices, thyme, dried",
#         "spices, pepper, black"
#       ],
#       "quantities": [
#         "12",
#         "1",
#         "1",
#         "2",
#         "1 1/4",
#         "14",
#         "1",
#         "1 1/2",
#         "12",
#         "1",
#         "1",
#         "12",
#         "3"
#       ],
#       "units": [
#         "cup",
#         "cup",
#         "tablespoon",
#         "tablespoon",
#         "cup",
#         "cup",
#         "teaspoon",
#         "teaspoon",
#         "teaspoon",
#         "teaspoon",
#         "teaspoon",
#         "teaspoon",
#         "dash"
#       ],
#       "meal_type": "dinner",
#       "recipe_id": "1d66c87df7",
#       "query": "vegetarian chickpea spinach curry dinner"
#     },
#     {
#       "title": "Soy Lime Roasted Tofu",
#       "calories": 932.40967,
#       "protein_g": 109.6122386,
#       "fat_g": 16.53537135,
#       "carbs_g": 30.02768415,
#       "description": "Tofu marinated in a mixture of soy sauce, lime juice and sesame oil, then roasted until golden brown; easy, flavorful vegan main dish.",
#       "instructions": "Pat tofu dry and cut into 1/2- to 3/4 inch cubes.\nCombine soy sauce, lime juice and oil in a medium shallow dish or large sealable plastic bag.\nAdd the tofu; gently toss to combine.\nMarinate in the refrigerator for 1 hour or up to 4 hours, gently stirring once or twice.\nPreheat oven to 450F.\nRemove the tofu from the marinade with a slotted spoon (discard marinade).\nSpread out on a large baking sheet, making sure the pieces are not touching.\nRoast, gently turning halfway through, until golden brown, about 20 minutes.",
#       "ingredients": [
#         "tofu, raw, regular, prepared with calcium sulfate",
#         "soy sauce made from soy (tamari)",
#         "lime juice, raw",
#         "oil, sesame, salad or cooking"
#       ],
#       "quantities": [
#         "14",
#         "13",
#         "13",
#         "3"
#       ],
#       "units": [
#         "ounce",
#         "cup",
#         "cup",
#         "tablespoon"
#       ],
#       "meal_type": "dinner",
#       "recipe_id": "f00ec9f59a",
#       "query": "vegetarian roasted broccoli tofu stir-fry dinner"
#     },
#     {
#       "title": "Basic Marinated and Baked Tofu",
#       "calories": 1036.2433066667,
#       "protein_g": 153.3367445333,
#       "fat_g": 17.7872325333,
#       "carbs_g": 29.4174234667,
#       "description": "Marinated tofu baked in the oven with a savory, slightly sweet sauce; simple, versatile vegan main dish.",
#       "instructions": "Wrap the tofu in paper towels and press under a heavy skillet for 15 minutes.\nIf it's still pretty wet, replace the towels and press it again.\nMix the soy sauce, vinegar, and oil in a container that is small but big enough for all the tofu.\nCut the tofu in half inch slices and place in marinade.\nTurn it over after about 15 minutes.\nPreheat oven to 350F.\nWipe some of the oil from the marinade on your baking sheet.\nArrange the tofu slices in one layer and bake for 15 minutes.\nTurn them over and bake another 15 minutes.",
#       "ingredients": [
#         "tofu, raw, regular, prepared with calcium sulfate",
#         "soy sauce made from soy (tamari)",
#         "rice vinegar, upc: 4979435030332",
#         "oil, sesame, salad or cooking"
#       ],
#       "quantities": [
#         "16",
#         "14",
#         "2",
#         "2"
#       ],
#       "units": [
#         "ounce",
#         "cup",
#         "tablespoon",
#         "tablespoon"
#       ],
#       "meal_type": "dinner",
#       "recipe_id": "59b78badd7",
#       "query": "vegetarian roasted broccoli tofu stir-fry dinner"
#     },
#     {
#       "title": "Vegan Soy, Lentil, & Veggie Burger Sliders Using Tvp",
#       "calories": 889.609375,
#       "protein_g": 46.81171875,
#       "fat_g": 21.841796875,
#       "carbs_g": 15.588984375,
#       "description": "Vegan burgers made with TVP, lentils, and vegetables, bound with flaxseed and baked into crispy patties; versatile and customizable.",
#       "instructions": "If you're using dry lentils on the stovetop and not the pressure cooker, bring 1/2 cup water to a boil, add the lentils, and lower the heat to medium and let simmer for 20 minutes; drain any excess water when done cooking.\nNuke the vegetable broth for about 2 minutes depending on your microwave's strength.\nYou want it to be near-boiling but not too hot to handle.\nAdd all the dry spices listed with the nutritional yeast under \"Base For The Sliders\" to the heated vegetable broth.\nPour into a large mixing bowl.\nAdd the TVP, making sure it's all covered.\nLet it sit for 10 minutes in order to expand, hydrate, and soak up all the seasonings.\nWhile you wait for that, finely shred the carrot if not using pre-shredded.\nIf not using pre-chopped broccoli like those frozen packages, finely chop up about 1/3 head of a broccoli to get about 1/2 cup of finely chopped broccoli pieces.\nPut the broccoli aside for now.\nHeat up the olive oil in a skillet and sautee the onion, garlic, carrot, and broccoli together until nicely browned, about 5 minutes.\nThe TVP should be all soaked up by now.\nStir it to make sure all the seasoning dispersed.\nAdd the lentils and sauteed veggies.\nMix thoroughly.\nBind with the flax seed.\nFeel free to add some bread crumbs for extra crunch, or more flax if it's not binding well enough.\nThis part was tricky for me because the original mix didn't bind at all.\nI don't recommend using eggs or egg whites to bind because it'll make the mix even more liquidy.\nPreheat the oven to 425F Put a little olive oil on a cookie sheet-- cooking spray just won't work the same for this-- about a teaspoon or a little less, and brush it on evenly with a BBQ/pastry brush or paper towel.\nWhile the oven preheats, let the mixture sit out for about 10 minutes.\nEtract the mixture and put it on the cookie sheet.\nCarefully squish it out evenly, sort of like you're stretching out pizza dough.\nIt should be a large thin rectangle.\nBake for 10-12 minutes or until golden brown on top.\nFlip it over so the other side can cook for another 10-12 minutes.\nUse a pizza cutter to cut into slider squares, you'll get about 16.\nServe on wheat rolls with lettuce, tomato, Vegenaise, Soy Kaas, avocado slices, and/or your other favorite vegetarian \"burger\" fixings!\nIf you want to make these as normal veggie burgers, you'll get 4 big patties and have to bake for about 15 minutes on each side.",
#       "ingredients": [
#         "shortening, vegetable, household, composite",
#         "soup, vegetable broth, ready to serve",
#         "leavening agents, yeast, baker's, active dry",
#         "spices, paprika",
#         "salt, table",
#         "spices, marjoram, dried",
#         "spices, onion powder",
#         "spices, pepper, black",
#         "mustard, prepared, yellow",
#         "spices, pepper, red or cayenne",
#         "onions, raw",
#         "spices, garlic powder",
#         "oil, olive, salad or cooking",
#         "broccoli, raw",
#         "carrots, raw",
#         "lentils, raw",
#         "seeds, flaxseed"
#       ],
#       "quantities": [
#         "1",
#         "1",
#         "2",
#         "14",
#         "14",
#         "14",
#         "18",
#         "18",
#         "18",
#         "1",
#         "14",
#         "12",
#         "2",
#         "12",
#         "12",
#         "13",
#         "1"
#       ],
#       "units": [
#         "cup",
#         "cup",
#         "tablespoon",
#         "teaspoon",
#         "teaspoon",
#         "teaspoon",
#         "teaspoon",
#         "teaspoon",
#         "teaspoon",
#         "dash",
#         "cup",
#         "teaspoon",
#         "tablespoon",
#         "cup",
#         "cup",
#         "cup",
#         "cup"
#       ],
#       "meal_type": "dinner",
#       "recipe_id": "266444cabe",
#       "query": "vegetarian chickpea vegetable tagine dinner"
#     },
#     {
#       "title": "Ppk's Simple Italian Sausages- Vegan/Vegetarian",
#       "calories": 1244.34375,
#       "protein_g": 154.10875,
#       "fat_g": 28.26265625,
#       "carbs_g": 5.48046875,
#       "description": "Vegan Italian sausages made with vital wheat gluten, beans, and spices, steamed to perfection; easy, plant-based alternative to traditional sausages.",
#       "instructions": "Get your tinfoil squares ready.\nIn a large bowl, mash the beans until no whole ones are left.\nThrow all the other ingredients together in the order listed and mix with a fork (I find it easier to use my hands).\nGet your steaming apparatus ready.\nDivide dough into 8 even parts and form into little logs (about the size of a large hot dog.\n).\nTightly wrap the dough in tin foil, like a tootsie roll, twisting the ends.\n(I didn't wrap mine tight enough last time and they came out strangely shaped but they still cooked through and tasted great.\n).\nPlace wrapped sausages in steamer (I use one of those cheap steamer baskets you can buy at the grocery store) and steam for 40 minutes.\nUnwrap and enjoy or refrigerate for later (they also freeze well).",
#       "ingredients": [
#         "beans, snap, green, raw",
#         "soup, vegetable broth, ready to serve",
#         "oil, olive, salad or cooking",
#         "soy sauce made from soy (tamari)",
#         "vital wheat gluten",
#         "leavening agents, yeast, baker's, active dry",
#         "spices, garlic powder",
#         "spices, fennel seed",
#         "spices, pepper, red or cayenne",
#         "spices, paprika",
#         "spices, oregano, dried",
#         "spices, thyme, dried",
#         "spices, pepper, black"
#       ],
#       "quantities": [
#         "12",
#         "1",
#         "1",
#         "2",
#         "1 1/4",
#         "14",
#         "1",
#         "1 1/2",
#         "12",
#         "1",
#         "1",
#         "12",
#         "3"
#       ],
#       "units": [
#         "cup",
#         "cup",
#         "tablespoon",
#         "tablespoon",
#         "cup",
#         "cup",
#         "teaspoon",
#         "teaspoon",
#         "teaspoon",
#         "teaspoon",
#         "teaspoon",
#         "teaspoon",
#         "dash"
#       ],
#       "meal_type": "dinner",
#       "recipe_id": "1d66c87df7",
#       "query": "vegetarian chickpea vegetable tagine dinner"
#     },
#     {
#       "title": "Ppk's Simple Italian Sausages- Vegan/Vegetarian",
#       "calories": 1244.34375,
#       "protein_g": 154.10875,
#       "fat_g": 28.26265625,
#       "carbs_g": 5.48046875,
#       "description": "Vegan Italian sausages made with vital wheat gluten, beans, and spices, steamed to perfection; easy, plant-based alternative to traditional sausages.",
#       "instructions": "Get your tinfoil squares ready.\nIn a large bowl, mash the beans until no whole ones are left.\nThrow all the other ingredients together in the order listed and mix with a fork (I find it easier to use my hands).\nGet your steaming apparatus ready.\nDivide dough into 8 even parts and form into little logs (about the size of a large hot dog.\n).\nTightly wrap the dough in tin foil, like a tootsie roll, twisting the ends.\n(I didn't wrap mine tight enough last time and they came out strangely shaped but they still cooked through and tasted great.\n).\nPlace wrapped sausages in steamer (I use one of those cheap steamer baskets you can buy at the grocery store) and steam for 40 minutes.\nUnwrap and enjoy or refrigerate for later (they also freeze well).",
#       "ingredients": [
#         "beans, snap, green, raw",
#         "soup, vegetable broth, ready to serve",
#         "oil, olive, salad or cooking",
#         "soy sauce made from soy (tamari)",
#         "vital wheat gluten",
#         "leavening agents, yeast, baker's, active dry",
#         "spices, garlic powder",
#         "spices, fennel seed",
#         "spices, pepper, red or cayenne",
#         "spices, paprika",
#         "spices, oregano, dried",
#         "spices, thyme, dried",
#         "spices, pepper, black"
#       ],
#       "quantities": [
#         "12",
#         "1",
#         "1",
#         "2",
#         "1 1/4",
#         "14",
#         "1",
#         "1 1/2",
#         "12",
#         "1",
#         "1",
#         "12",
#         "3"
#       ],
#       "units": [
#         "cup",
#         "cup",
#         "tablespoon",
#         "tablespoon",
#         "cup",
#         "cup",
#         "teaspoon",
#         "teaspoon",
#         "teaspoon",
#         "teaspoon",
#         "teaspoon",
#         "teaspoon",
#         "dash"
#       ],
#       "meal_type": "dinner",
#       "recipe_id": "1d66c87df7",
#       "query": "vegetarian broccoli cheese casserole dinner"
#     },
#     {
#       "title": "Vegan Soy, Lentil, & Veggie Burger Sliders Using Tvp",
#       "calories": 889.609375,
#       "protein_g": 46.81171875,
#       "fat_g": 21.841796875,
#       "carbs_g": 15.588984375,
#       "description": "Vegan burgers made with TVP, lentils, and vegetables, bound with flaxseed and baked into crispy patties; versatile and customizable.",
#       "instructions": "If you're using dry lentils on the stovetop and not the pressure cooker, bring 1/2 cup water to a boil, add the lentils, and lower the heat to medium and let simmer for 20 minutes; drain any excess water when done cooking.\nNuke the vegetable broth for about 2 minutes depending on your microwave's strength.\nYou want it to be near-boiling but not too hot to handle.\nAdd all the dry spices listed with the nutritional yeast under \"Base For The Sliders\" to the heated vegetable broth.\nPour into a large mixing bowl.\nAdd the TVP, making sure it's all covered.\nLet it sit for 10 minutes in order to expand, hydrate, and soak up all the seasonings.\nWhile you wait for that, finely shred the carrot if not using pre-shredded.\nIf not using pre-chopped broccoli like those frozen packages, finely chop up about 1/3 head of a broccoli to get about 1/2 cup of finely chopped broccoli pieces.\nPut the broccoli aside for now.\nHeat up the olive oil in a skillet and sautee the onion, garlic, carrot, and broccoli together until nicely browned, about 5 minutes.\nThe TVP should be all soaked up by now.\nStir it to make sure all the seasoning dispersed.\nAdd the lentils and sauteed veggies.\nMix thoroughly.\nBind with the flax seed.\nFeel free to add some bread crumbs for extra crunch, or more flax if it's not binding well enough.\nThis part was tricky for me because the original mix didn't bind at all.\nI don't recommend using eggs or egg whites to bind because it'll make the mix even more liquidy.\nPreheat the oven to 425F Put a little olive oil on a cookie sheet-- cooking spray just won't work the same for this-- about a teaspoon or a little less, and brush it on evenly with a BBQ/pastry brush or paper towel.\nWhile the oven preheats, let the mixture sit out for about 10 minutes.\nEtract the mixture and put it on the cookie sheet.\nCarefully squish it out evenly, sort of like you're stretching out pizza dough.\nIt should be a large thin rectangle.\nBake for 10-12 minutes or until golden brown on top.\nFlip it over so the other side can cook for another 10-12 minutes.\nUse a pizza cutter to cut into slider squares, you'll get about 16.\nServe on wheat rolls with lettuce, tomato, Vegenaise, Soy Kaas, avocado slices, and/or your other favorite vegetarian \"burger\" fixings!\nIf you want to make these as normal veggie burgers, you'll get 4 big patties and have to bake for about 15 minutes on each side.",
#       "ingredients": [
#         "shortening, vegetable, household, composite",
#         "soup, vegetable broth, ready to serve",
#         "leavening agents, yeast, baker's, active dry",
#         "spices, paprika",
#         "salt, table",
#         "spices, marjoram, dried",
#         "spices, onion powder",
#         "spices, pepper, black",
#         "mustard, prepared, yellow",
#         "spices, pepper, red or cayenne",
#         "onions, raw",
#         "spices, garlic powder",
#         "oil, olive, salad or cooking",
#         "broccoli, raw",
#         "carrots, raw",
#         "lentils, raw",
#         "seeds, flaxseed"
#       ],
#       "quantities": [
#         "1",
#         "1",
#         "2",
#         "14",
#         "14",
#         "14",
#         "18",
#         "18",
#         "18",
#         "1",
#         "14",
#         "12",
#         "2",
#         "12",
#         "12",
#         "13",
#         "1"
#       ],
#       "units": [
#         "cup",
#         "cup",
#         "tablespoon",
#         "teaspoon",
#         "teaspoon",
#         "teaspoon",
#         "teaspoon",
#         "teaspoon",
#         "teaspoon",
#         "dash",
#         "cup",
#         "teaspoon",
#         "tablespoon",
#         "cup",
#         "cup",
#         "cup",
#         "cup"
#       ],
#       "meal_type": "dinner",
#       "recipe_id": "266444cabe",
#       "query": "vegetarian broccoli cheese casserole dinner"
#     },
#     {
#       "title": "Vegan Soy, Lentil, & Veggie Burger Sliders Using Tvp",
#       "calories": 889.609375,
#       "protein_g": 46.81171875,
#       "fat_g": 21.841796875,
#       "carbs_g": 15.588984375,
#       "description": "Vegan burgers made with TVP, lentils, and vegetables, bound with flaxseed and baked into crispy patties; versatile and customizable.",
#       "instructions": "If you're using dry lentils on the stovetop and not the pressure cooker, bring 1/2 cup water to a boil, add the lentils, and lower the heat to medium and let simmer for 20 minutes; drain any excess water when done cooking.\nNuke the vegetable broth for about 2 minutes depending on your microwave's strength.\nYou want it to be near-boiling but not too hot to handle.\nAdd all the dry spices listed with the nutritional yeast under \"Base For The Sliders\" to the heated vegetable broth.\nPour into a large mixing bowl.\nAdd the TVP, making sure it's all covered.\nLet it sit for 10 minutes in order to expand, hydrate, and soak up all the seasonings.\nWhile you wait for that, finely shred the carrot if not using pre-shredded.\nIf not using pre-chopped broccoli like those frozen packages, finely chop up about 1/3 head of a broccoli to get about 1/2 cup of finely chopped broccoli pieces.\nPut the broccoli aside for now.\nHeat up the olive oil in a skillet and sautee the onion, garlic, carrot, and broccoli together until nicely browned, about 5 minutes.\nThe TVP should be all soaked up by now.\nStir it to make sure all the seasoning dispersed.\nAdd the lentils and sauteed veggies.\nMix thoroughly.\nBind with the flax seed.\nFeel free to add some bread crumbs for extra crunch, or more flax if it's not binding well enough.\nThis part was tricky for me because the original mix didn't bind at all.\nI don't recommend using eggs or egg whites to bind because it'll make the mix even more liquidy.\nPreheat the oven to 425F Put a little olive oil on a cookie sheet-- cooking spray just won't work the same for this-- about a teaspoon or a little less, and brush it on evenly with a BBQ/pastry brush or paper towel.\nWhile the oven preheats, let the mixture sit out for about 10 minutes.\nEtract the mixture and put it on the cookie sheet.\nCarefully squish it out evenly, sort of like you're stretching out pizza dough.\nIt should be a large thin rectangle.\nBake for 10-12 minutes or until golden brown on top.\nFlip it over so the other side can cook for another 10-12 minutes.\nUse a pizza cutter to cut into slider squares, you'll get about 16.\nServe on wheat rolls with lettuce, tomato, Vegenaise, Soy Kaas, avocado slices, and/or your other favorite vegetarian \"burger\" fixings!\nIf you want to make these as normal veggie burgers, you'll get 4 big patties and have to bake for about 15 minutes on each side.",
#       "ingredients": [
#         "shortening, vegetable, household, composite",
#         "soup, vegetable broth, ready to serve",
#         "leavening agents, yeast, baker's, active dry",
#         "spices, paprika",
#         "salt, table",
#         "spices, marjoram, dried",
#         "spices, onion powder",
#         "spices, pepper, black",
#         "mustard, prepared, yellow",
#         "spices, pepper, red or cayenne",
#         "onions, raw",
#         "spices, garlic powder",
#         "oil, olive, salad or cooking",
#         "broccoli, raw",
#         "carrots, raw",
#         "lentils, raw",
#         "seeds, flaxseed"
#       ],
#       "quantities": [
#         "1",
#         "1",
#         "2",
#         "14",
#         "14",
#         "14",
#         "18",
#         "18",
#         "18",
#         "1",
#         "14",
#         "12",
#         "2",
#         "12",
#         "12",
#         "13",
#         "1"
#       ],
#       "units": [
#         "cup",
#         "cup",
#         "tablespoon",
#         "teaspoon",
#         "teaspoon",
#         "teaspoon",
#         "teaspoon",
#         "teaspoon",
#         "teaspoon",
#         "dash",
#         "cup",
#         "teaspoon",
#         "tablespoon",
#         "cup",
#         "cup",
#         "cup",
#         "cup"
#       ],
#       "meal_type": "dinner",
#       "recipe_id": "266444cabe",
#       "query": "vegetarian chickpea sweet potato curry dinner"
#     },
#     {
#       "title": "Ppk's Simple Italian Sausages- Vegan/Vegetarian",
#       "calories": 1244.34375,
#       "protein_g": 154.10875,
#       "fat_g": 28.26265625,
#       "carbs_g": 5.48046875,
#       "description": "Vegan Italian sausages made with vital wheat gluten, beans, and spices, steamed to perfection; easy, plant-based alternative to traditional sausages.",
#       "instructions": "Get your tinfoil squares ready.\nIn a large bowl, mash the beans until no whole ones are left.\nThrow all the other ingredients together in the order listed and mix with a fork (I find it easier to use my hands).\nGet your steaming apparatus ready.\nDivide dough into 8 even parts and form into little logs (about the size of a large hot dog.\n).\nTightly wrap the dough in tin foil, like a tootsie roll, twisting the ends.\n(I didn't wrap mine tight enough last time and they came out strangely shaped but they still cooked through and tasted great.\n).\nPlace wrapped sausages in steamer (I use one of those cheap steamer baskets you can buy at the grocery store) and steam for 40 minutes.\nUnwrap and enjoy or refrigerate for later (they also freeze well).",
#       "ingredients": [
#         "beans, snap, green, raw",
#         "soup, vegetable broth, ready to serve",
#         "oil, olive, salad or cooking",
#         "soy sauce made from soy (tamari)",
#         "vital wheat gluten",
#         "leavening agents, yeast, baker's, active dry",
#         "spices, garlic powder",
#         "spices, fennel seed",
#         "spices, pepper, red or cayenne",
#         "spices, paprika",
#         "spices, oregano, dried",
#         "spices, thyme, dried",
#         "spices, pepper, black"
#       ],
#       "quantities": [
#         "12",
#         "1",
#         "1",
#         "2",
#         "1 1/4",
#         "14",
#         "1",
#         "1 1/2",
#         "12",
#         "1",
#         "1",
#         "12",
#         "3"
#       ],
#       "units": [
#         "cup",
#         "cup",
#         "tablespoon",
#         "tablespoon",
#         "cup",
#         "cup",
#         "teaspoon",
#         "teaspoon",
#         "teaspoon",
#         "teaspoon",
#         "teaspoon",
#         "teaspoon",
#         "dash"
#       ],
#       "meal_type": "dinner",
#       "recipe_id": "1d66c87df7",
#       "query": "vegetarian chickpea sweet potato curry dinner"
#     },
#     {
#       "title": "Soy Lime Roasted Tofu",
#       "calories": 932.40967,
#       "protein_g": 109.6122386,
#       "fat_g": 16.53537135,
#       "carbs_g": 30.02768415,
#       "description": "Tofu marinated in a mixture of soy sauce, lime juice and sesame oil, then roasted until golden brown; easy, flavorful vegan main dish.",
#       "instructions": "Pat tofu dry and cut into 1/2- to 3/4 inch cubes.\nCombine soy sauce, lime juice and oil in a medium shallow dish or large sealable plastic bag.\nAdd the tofu; gently toss to combine.\nMarinate in the refrigerator for 1 hour or up to 4 hours, gently stirring once or twice.\nPreheat oven to 450F.\nRemove the tofu from the marinade with a slotted spoon (discard marinade).\nSpread out on a large baking sheet, making sure the pieces are not touching.\nRoast, gently turning halfway through, until golden brown, about 20 minutes.",
#       "ingredients": [
#         "tofu, raw, regular, prepared with calcium sulfate",
#         "soy sauce made from soy (tamari)",
#         "lime juice, raw",
#         "oil, sesame, salad or cooking"
#       ],
#       "quantities": [
#         "14",
#         "13",
#         "13",
#         "3"
#       ],
#       "units": [
#         "ounce",
#         "cup",
#         "cup",
#         "tablespoon"
#       ],
#       "meal_type": "dinner",
#       "recipe_id": "f00ec9f59a",
#       "query": "vegetarian roasted broccoli cauliflower dinner"
#     },
#     {
#       "title": "Vegan Soy, Lentil, & Veggie Burger Sliders Using Tvp",
#       "calories": 889.609375,
#       "protein_g": 46.81171875,
#       "fat_g": 21.841796875,
#       "carbs_g": 15.588984375,
#       "description": "Vegan burgers made with TVP, lentils, and vegetables, bound with flaxseed and baked into crispy patties; versatile and customizable.",
#       "instructions": "If you're using dry lentils on the stovetop and not the pressure cooker, bring 1/2 cup water to a boil, add the lentils, and lower the heat to medium and let simmer for 20 minutes; drain any excess water when done cooking.\nNuke the vegetable broth for about 2 minutes depending on your microwave's strength.\nYou want it to be near-boiling but not too hot to handle.\nAdd all the dry spices listed with the nutritional yeast under \"Base For The Sliders\" to the heated vegetable broth.\nPour into a large mixing bowl.\nAdd the TVP, making sure it's all covered.\nLet it sit for 10 minutes in order to expand, hydrate, and soak up all the seasonings.\nWhile you wait for that, finely shred the carrot if not using pre-shredded.\nIf not using pre-chopped broccoli like those frozen packages, finely chop up about 1/3 head of a broccoli to get about 1/2 cup of finely chopped broccoli pieces.\nPut the broccoli aside for now.\nHeat up the olive oil in a skillet and sautee the onion, garlic, carrot, and broccoli together until nicely browned, about 5 minutes.\nThe TVP should be all soaked up by now.\nStir it to make sure all the seasoning dispersed.\nAdd the lentils and sauteed veggies.\nMix thoroughly.\nBind with the flax seed.\nFeel free to add some bread crumbs for extra crunch, or more flax if it's not binding well enough.\nThis part was tricky for me because the original mix didn't bind at all.\nI don't recommend using eggs or egg whites to bind because it'll make the mix even more liquidy.\nPreheat the oven to 425F Put a little olive oil on a cookie sheet-- cooking spray just won't work the same for this-- about a teaspoon or a little less, and brush it on evenly with a BBQ/pastry brush or paper towel.\nWhile the oven preheats, let the mixture sit out for about 10 minutes.\nEtract the mixture and put it on the cookie sheet.\nCarefully squish it out evenly, sort of like you're stretching out pizza dough.\nIt should be a large thin rectangle.\nBake for 10-12 minutes or until golden brown on top.\nFlip it over so the other side can cook for another 10-12 minutes.\nUse a pizza cutter to cut into slider squares, you'll get about 16.\nServe on wheat rolls with lettuce, tomato, Vegenaise, Soy Kaas, avocado slices, and/or your other favorite vegetarian \"burger\" fixings!\nIf you want to make these as normal veggie burgers, you'll get 4 big patties and have to bake for about 15 minutes on each side.",
#       "ingredients": [
#         "shortening, vegetable, household, composite",
#         "soup, vegetable broth, ready to serve",
#         "leavening agents, yeast, baker's, active dry",
#         "spices, paprika",
#         "salt, table",
#         "spices, marjoram, dried",
#         "spices, onion powder",
#         "spices, pepper, black",
#         "mustard, prepared, yellow",
#         "spices, pepper, red or cayenne",
#         "onions, raw",
#         "spices, garlic powder",
#         "oil, olive, salad or cooking",
#         "broccoli, raw",
#         "carrots, raw",
#         "lentils, raw",
#         "seeds, flaxseed"
#       ],
#       "quantities": [
#         "1",
#         "1",
#         "2",
#         "14",
#         "14",
#         "14",
#         "18",
#         "18",
#         "18",
#         "1",
#         "14",
#         "12",
#         "2",
#         "12",
#         "12",
#         "13",
#         "1"
#       ],
#       "units": [
#         "cup",
#         "cup",
#         "tablespoon",
#         "teaspoon",
#         "teaspoon",
#         "teaspoon",
#         "teaspoon",
#         "teaspoon",
#         "teaspoon",
#         "dash",
#         "cup",
#         "teaspoon",
#         "tablespoon",
#         "cup",
#         "cup",
#         "cup",
#         "cup"
#       ],
#       "meal_type": "dinner",
#       "recipe_id": "266444cabe",
#       "query": "vegetarian roasted broccoli cauliflower dinner"
#     },
#     {
#       "title": "Vegan Soy, Lentil, & Veggie Burger Sliders Using Tvp",
#       "calories": 889.609375,
#       "protein_g": 46.81171875,
#       "fat_g": 21.841796875,
#       "carbs_g": 15.588984375,
#       "description": "Vegan burgers made with TVP, lentils, and vegetables, bound with flaxseed and baked into crispy patties; versatile and customizable.",
#       "instructions": "If you're using dry lentils on the stovetop and not the pressure cooker, bring 1/2 cup water to a boil, add the lentils, and lower the heat to medium and let simmer for 20 minutes; drain any excess water when done cooking.\nNuke the vegetable broth for about 2 minutes depending on your microwave's strength.\nYou want it to be near-boiling but not too hot to handle.\nAdd all the dry spices listed with the nutritional yeast under \"Base For The Sliders\" to the heated vegetable broth.\nPour into a large mixing bowl.\nAdd the TVP, making sure it's all covered.\nLet it sit for 10 minutes in order to expand, hydrate, and soak up all the seasonings.\nWhile you wait for that, finely shred the carrot if not using pre-shredded.\nIf not using pre-chopped broccoli like those frozen packages, finely chop up about 1/3 head of a broccoli to get about 1/2 cup of finely chopped broccoli pieces.\nPut the broccoli aside for now.\nHeat up the olive oil in a skillet and sautee the onion, garlic, carrot, and broccoli together until nicely browned, about 5 minutes.\nThe TVP should be all soaked up by now.\nStir it to make sure all the seasoning dispersed.\nAdd the lentils and sauteed veggies.\nMix thoroughly.\nBind with the flax seed.\nFeel free to add some bread crumbs for extra crunch, or more flax if it's not binding well enough.\nThis part was tricky for me because the original mix didn't bind at all.\nI don't recommend using eggs or egg whites to bind because it'll make the mix even more liquidy.\nPreheat the oven to 425F Put a little olive oil on a cookie sheet-- cooking spray just won't work the same for this-- about a teaspoon or a little less, and brush it on evenly with a BBQ/pastry brush or paper towel.\nWhile the oven preheats, let the mixture sit out for about 10 minutes.\nEtract the mixture and put it on the cookie sheet.\nCarefully squish it out evenly, sort of like you're stretching out pizza dough.\nIt should be a large thin rectangle.\nBake for 10-12 minutes or until golden brown on top.\nFlip it over so the other side can cook for another 10-12 minutes.\nUse a pizza cutter to cut into slider squares, you'll get about 16.\nServe on wheat rolls with lettuce, tomato, Vegenaise, Soy Kaas, avocado slices, and/or your other favorite vegetarian \"burger\" fixings!\nIf you want to make these as normal veggie burgers, you'll get 4 big patties and have to bake for about 15 minutes on each side.",
#       "ingredients": [
#         "shortening, vegetable, household, composite",
#         "soup, vegetable broth, ready to serve",
#         "leavening agents, yeast, baker's, active dry",
#         "spices, paprika",
#         "salt, table",
#         "spices, marjoram, dried",
#         "spices, onion powder",
#         "spices, pepper, black",
#         "mustard, prepared, yellow",
#         "spices, pepper, red or cayenne",
#         "onions, raw",
#         "spices, garlic powder",
#         "oil, olive, salad or cooking",
#         "broccoli, raw",
#         "carrots, raw",
#         "lentils, raw",
#         "seeds, flaxseed"
#       ],
#       "quantities": [
#         "1",
#         "1",
#         "2",
#         "14",
#         "14",
#         "14",
#         "18",
#         "18",
#         "18",
#         "1",
#         "14",
#         "12",
#         "2",
#         "12",
#         "12",
#         "13",
#         "1"
#       ],
#       "units": [
#         "cup",
#         "cup",
#         "tablespoon",
#         "teaspoon",
#         "teaspoon",
#         "teaspoon",
#         "teaspoon",
#         "teaspoon",
#         "teaspoon",
#         "dash",
#         "cup",
#         "teaspoon",
#         "tablespoon",
#         "cup",
#         "cup",
#         "cup",
#         "cup"
#       ],
#       "meal_type": "dinner",
#       "recipe_id": "266444cabe",
#       "query": "vegetarian chickpea zucchini stew dinner"
#     },
#     {
#       "title": "Ppk's Simple Italian Sausages- Vegan/Vegetarian",
#       "calories": 1244.34375,
#       "protein_g": 154.10875,
#       "fat_g": 28.26265625,
#       "carbs_g": 5.48046875,
#       "description": "Vegan Italian sausages made with vital wheat gluten, beans, and spices, steamed to perfection; easy, plant-based alternative to traditional sausages.",
#       "instructions": "Get your tinfoil squares ready.\nIn a large bowl, mash the beans until no whole ones are left.\nThrow all the other ingredients together in the order listed and mix with a fork (I find it easier to use my hands).\nGet your steaming apparatus ready.\nDivide dough into 8 even parts and form into little logs (about the size of a large hot dog.\n).\nTightly wrap the dough in tin foil, like a tootsie roll, twisting the ends.\n(I didn't wrap mine tight enough last time and they came out strangely shaped but they still cooked through and tasted great.\n).\nPlace wrapped sausages in steamer (I use one of those cheap steamer baskets you can buy at the grocery store) and steam for 40 minutes.\nUnwrap and enjoy or refrigerate for later (they also freeze well).",
#       "ingredients": [
#         "beans, snap, green, raw",
#         "soup, vegetable broth, ready to serve",
#         "oil, olive, salad or cooking",
#         "soy sauce made from soy (tamari)",
#         "vital wheat gluten",
#         "leavening agents, yeast, baker's, active dry",
#         "spices, garlic powder",
#         "spices, fennel seed",
#         "spices, pepper, red or cayenne",
#         "spices, paprika",
#         "spices, oregano, dried",
#         "spices, thyme, dried",
#         "spices, pepper, black"
#       ],
#       "quantities": [
#         "12",
#         "1",
#         "1",
#         "2",
#         "1 1/4",
#         "14",
#         "1",
#         "1 1/2",
#         "12",
#         "1",
#         "1",
#         "12",
#         "3"
#       ],
#       "units": [
#         "cup",
#         "cup",
#         "tablespoon",
#         "tablespoon",
#         "cup",
#         "cup",
#         "teaspoon",
#         "teaspoon",
#         "teaspoon",
#         "teaspoon",
#         "teaspoon",
#         "teaspoon",
#         "dash"
#       ],
#       "meal_type": "dinner",
#       "recipe_id": "1d66c87df7",
#       "query": "vegetarian chickpea zucchini stew dinner"
#     }
#   ],
#   "original_request": {
#     "target_calories": 2500,
#     "target_protein": 100,
#     "target_fat": 60,
#     "target_carbs": 200,
#     "breakfast_targets": {
#       "macro_target_pct": 30,
#       "macro_tolerance_pct": 10
#     },
#     "lunch_targets": {
#       "macro_target_pct": 30,
#       "macro_tolerance_pct": 10
#     },
#     "dinner_targets": {
#       "macro_target_pct": 40,
#       "macro_tolerance_pct": 10
#     },
#     "dietary": [
#       "vegetarian"
#     ],
#     "num_days": 7,
#     "queries_per_meal": 7,
#     "candidate_recipes": 42,
#     "preferences": "I want a mix of pancakes and oatmeal for breakfast, some salads for lunch, and hearty dinners with chickpeas or broccoli",
#     "exclusions": "peanuts, mushrooms"
#   },
#   "queries": [
#     "vegetarian blueberry oatmeal pancakes breakfast",
#     "vegetarian greek salad feta lunch",
#     "vegetarian chickpea spinach curry dinner",
#     "vegetarian banana oatmeal breakfast bowl",
#     "vegetarian quinoa avocado salad lunch",
#     "vegetarian roasted broccoli tofu stir-fry dinner",
#     "vegetarian apple cinnamon oatmeal breakfast",
#     "vegetarian caprese salad balsamic glaze lunch",
#     "vegetarian chickpea vegetable tagine dinner",
#     "vegetarian chocolate chip pancakes breakfast",
#     "vegetarian kale quinoa salad lunch",
#     "vegetarian broccoli cheese casserole dinner",
#     "vegetarian strawberry oatmeal smoothie bowl breakfast",
#     "vegetarian mediterranean chickpea salad lunch",
#     "vegetarian chickpea sweet potato curry dinner",
#     "vegetarian peanut butter banana pancakes breakfast",
#     "vegetarian spinach strawberry salad lunch",
#     "vegetarian roasted broccoli cauliflower dinner",
#     "vegetarian maple walnut oatmeal breakfast",
#     "vegetarian tomato mozzarella salad lunch",
#     "vegetarian chickpea zucchini stew dinner"
#   ],
#   "meal_counts": {
#     "breakfast": 14,
#     "lunch": 14,
#     "dinner": 14
#   }
# }  

    

# For Debugging / Diagnostics
# diag = APIRouter()

# @diag.get("/_diag/env")
# def diag_env():
#     v = os.getenv("QDRANT_API_KEY", "")
#     return {
#         "QDRANT_URL": QDRANT_URL,
#         "QDRANT_COLLECTION": COLLECTION,
#         "QDRANT_API_KEY_present": bool(v),
#         "QDRANT_API_KEY_len": len(v or ""),
#         "QDRANT_API_KEY_is_json": (v.strip().startswith("{") if v else False),
#         "QDRANT_API_KEY_sample": (v.strip()[:4] + "..." + v.strip()[-4:] if v else ""),
#     }

# @diag.get("/_diag/qdrant")
# def diag_qdrant():
#     try:
#         # lightweight auth probe
#         r = client.get_collection(COLLECTION)
#         return {"ok": True, "collection": COLLECTION}
#     except Exception as e:
#         return {"ok": False, "error": f"{type(e).__name__}: {e}"}

# app.include_router(diag)
