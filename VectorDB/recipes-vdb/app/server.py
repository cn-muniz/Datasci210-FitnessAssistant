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
# logging.info(f"QDRANT_URL: {QDRANT_URL}\nQDRANT_API_KEY: {QDRANT_API_KEY}\n")

COHERE_API_BASE = os.getenv("COHERE_URL", "https://api.cohere.com/v2/chat")
COHERE_API_KEY = _get_env_key("COHERE_API_KEY")
# logging.info(f"COHERE_API_BASE: {COHERE_API_BASE}\nCOHERE_API_KEY: {COHERE_API_KEY}\n")

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
    logging.info(f"Received payload for n-day meal plan: {payload}")
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