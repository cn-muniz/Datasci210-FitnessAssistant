import datetime
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
# Show line number in logs
logging.getLogger().handlers[0].setFormatter(
    logging.Formatter(
        "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )
)
import llm_planning  # for meal plan prompt templates
import llm_judge
from grocery_consolidation import aggregate_items, create_grocery_prompt, parse_ndjson_response, flatten_ingredients, chunk_list
import meal_plan_evaluation  # for RAGAS evaluation of meal plans

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

# Suppress health check access logs while keeping all other requests
class HealthcheckFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        """
        uvicorn access logs carry the request line in either record.request_line
        or the formatted message itself; block entries that mention /status.
        """
        request_line = getattr(record, "request_line", "") or ""
        message = record.getMessage() if record.args else record.msg
        combined = f"{request_line} {message}"
        return "/status" not in combined

logging.getLogger("uvicorn.access").addFilter(HealthcheckFilter())

# ---------------- Config ----------------

# COLLECTION = os.getenv("QDRANT_COLLECTION", "recipes")
COLLECTION = os.getenv("QDRANT_COLLECTION", "recipes_v2")

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
    recipe_id: Optional[str] = Field(None, description="Unique ID from Qdrant vectorDB")
    merge_ingredients: Optional[List[dict]] = Field(None, description="Merged list of ingredients")
    data_source: Optional[str] = Field(None, description="Data source this recipe came from")

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

BREAKFAST_MACRO_DEFAULT = MealMacroTargets(macro_target_pct=25.0, macro_tolerance_pct=10.0)
LUNCH_MACRO_DEFAULT = MealMacroTargets(macro_target_pct=35.0, macro_tolerance_pct=10.0)
DINNER_MACRO_DEFAULT = MealMacroTargets(macro_target_pct=40.0, macro_tolerance_pct=10.0)

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
    merge_ingredients: Optional[List[dict]] = Field(None, description="Merged list of ingredients")
    data_source: Optional[str] = Field(None, description="Data source this recipe came from")

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

class GenerateShoppingListRequest(BaseModel):
    meal_descriptions: List[List[str]] = Field(..., description="List of lists of ingredient strings")
    model: Optional[str] = Field("command-a-03-2025", description="Cohere chat model to use")
    class Config:
        json_schema_extra = {
            "example": {
                "meal_descriptions": [
                    ["2 cups all purpose flour", "1.5 cups AP flour", "2 cups plain flour"],
                    ["3 scallions", "2 green onions", "4 large scallions"]
                ],
                "model": "command-a-03-2025"
            }
        }

class GenerateShoppingListResponse(BaseModel):
    shopping_list: List[Dict[str, Any]] = Field(..., description="Consolidated shopping list")
    notes: Optional[List[str]] = Field(None, description="Optional notes about the shopping list")

class MergeIngredients(BaseModel):
    o: str = Field(..., description="Original ingredient string")
    n: str = Field(..., description="Normalized ingredient name")
    q: float = Field(..., description="Quantity as a float")
    u: str = Field(..., description="Unit of measure")
    f: str = Field(..., description="Unit family ('mass', 'volume', or 'count')")
    c: str = Field(..., description="Grocery category")
    m: str = Field(..., description="Merge ingredient name")
    r: Optional[str] = Field(None, description="Optional raw unit text")
    ri: Optional[str] = Field(None, description="Optional raw ingredient text")
    data_source: Optional[str] = Field(None, description="Optional data source this ingredient came from")
    title: Optional[str] = Field(None, description="Optional recipe title this ingredient came from")
    recipe_id: Optional[str] = Field(None, description="Optional recipe ID this ingredient came from")

class GenerateShoppingListPreTaggedRequest(BaseModel):
    merge_ingredients: List[MergeIngredients] = Field(..., description="List of pre-tagged ingredients to merge")
    class Config:
        json_schema_extra = {
            "example": {
                "merge_ingredients": [
                    {"o": "all purpose flour", "n": "all purpose flour", "q": 2.0, "u": "cups", "f": "volume", "c": "Bread and Baked Goods", "m": "all purpose flour", "title": "Recipe A", "recipe_id": "abc123"},
                    {"o": "AP flour", "n": "all purpose flour", "q": 1.5, "u": "cups", "f": "volume", "c": "Bread and Baked Goods", "m": "all purpose flour", "title": "Recipe B", "recipe_id": "def456"},
                    {"o": "scallions", "n": "green onions", "q": 3.0, "u": "count", "f": "count", "c": "Vegetables", "m": "green onions", "title": "Recipe A", "recipe_id": "abc123"},
                    {"o": "green onions", "n": "green onions", "q": 2.0, "u": "count", "f": "count", "c": "Vegetables", "m": "green onions", "title": "Recipe C", "recipe_id": "ghi789"}
                ]
            }
        }    

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
        recipe_id=recipe_payload.get("id"),
        merge_ingredients=recipe_payload.get("merge_ingredients", []),
        data_source=recipe_payload.get("data_source", None)
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
        recipe_id=recipe_payload.get("recipe_id", None),
        merge_ingredients=recipe_payload.get("merge_ingredients", []),
        data_source=recipe_payload.get("data_source", None)
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
        
    # logging.info(recipe_payload)

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
        recipe_id=recipe_payload.get("id"),
        query=query,
        merge_ingredients=recipe_payload.get("merge_ingredients", []),
        data_source=recipe_payload.get("data_source", None)
    )

def _cohere_chat(messages, model="command-a-03-2025", json_enforce = False, timeout=45):
    url = COHERE_API_BASE
    # headers = {"Authorization": f"Bearer {COHERE_API_KEY}", "Content-Type": "application/json"}
    headers = {"Authorization": f"Bearer {COHERE_API_KEY}"}
    payload = {
        "model": model,
        "messages": messages,
        "stream": False
    }
    if json_enforce:
        payload['response_format'] = {"type": "json_object"}
    # logging.info(f"Sending request to Cohere chat model '{model}' with payload: {payload} to url: {url}")
    r = requests.post(url, json=payload, headers=headers, timeout=timeout)
    # print(items)
    # logging.info(f"Cohere response status: {r.status_code}, content: {r.text}")
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

    # First generate the n-day plan using the LLM
    if payload.num_days < 1 or payload.num_days > 7:
        raise HTTPException(status_code=400, detail={"message": "num_days must be between 1 and 7"})
    
    # Keep track of how long it takes to return the entire n-day plan
    start_total_time = time.time()

    logging.info(f"Received payload for n-day meal plan: {payload}")
    get_candidate_recipes_request = NDayRecipesRequest(
        target_calories=payload.target_calories,
        target_protein=payload.target_protein,
        target_fat=payload.target_fat,
        target_carbs=payload.target_carbs,
        dietary=payload.dietary,
        num_days=payload.num_days,
        queries_per_meal=7, # queries per meal type - hardcoded for now
        candidate_recipes=63,
        preferences=payload.preferences,
        exclusions=payload.exclusions
    )

    logging.info(f"Fetching candidate recipes with request: {get_candidate_recipes_request}")

    nday_recipes_response = get_candidate_recipes(get_candidate_recipes_request)
    # Print recipe titles and queries used
    # for recipe in nday_recipes_response.candidate_recipes:
    #     logging.info(f"Candidate recipe: {recipe.title} (meal_type={recipe.meal_type}, query='{recipe.query}')")

    # TODO: consider dropping quantities and units from candidates to free up more tokens
    candidate_recipes_full = nday_recipes_response.candidate_recipes
    # logging.info(candidate_recipes_full)
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
            "meal_type": recipe.meal_type,
            "data_source": recipe.data_source,
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

    # Save prompt to file for debugging
    # with open("llm_judge_prompt.json", "w") as f:
    #     f.write(judge_prompt[0]["content"])

    # Call the LLM
    try:
        # Get current time to measure how long it takes to get a response
        # logging.info(f"Calling LLM judge with prompt\n{judge_prompt}")
        logging.info(f"Calling LLM judge...")
        start_time = time.time()
        
        resp = _cohere_chat(judge_prompt, model="command-a-03-2025",json_enforce=True)
        end_time = time.time()
        duration = end_time - start_time
        logging.info(f"LLM judge call took {duration:.2f} seconds")
    except Exception as e:
        raise HTTPException(status_code=500, detail={"message": f"LLM call failed: {type(e).__name__}: {e}"})
    # logging.info(f"LLM response: \n{resp}")

    selected_recipes = llm_judge.extract_meal_plan_json(resp)

    ## Recipe ID mapping
    # Check to see if the recipe_ids are valid
    full_recipe_id_list = [a.recipe_id for a in candidate_recipes_full]

    daily_plans = []

    # for day in selected_recipes.get("days"):
    for k,day in selected_recipes.items():
        day_meals = {
            # "day": day.get("day"),
            "day": int(k.strip("day_")),
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
        for meal_key,recipe_id in day.items():
            if recipe_id not in full_recipe_id_list:
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
                    "recipe_id": recipe_id,
                    "merge_ingredients": [],
                    "data_source": "Error"
                })
            else:
                recipe_index = full_recipe_id_list.index(recipe_id)
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
    end_total_time = time.time()
    total_duration = end_total_time - start_total_time
    logging.info(f"Total n-day meal plan generation took {total_duration:.2f} seconds")
    
    # Evaluate the meal plan using RAGAS
    try:
        # Format meal plan for evaluation
        meal_plan_for_eval = {"days": daily_plans}
        
        # Format query for evaluation
        query_for_eval = {
            "num_days": payload.num_days,
            "target_calories": payload.target_calories,
            "target_protein": payload.target_protein,
            "target_fat": payload.target_fat,
            "target_carbs": payload.target_carbs,
            "exclusions": payload.exclusions or [],
            "dietary": payload.dietary or [],
            "preferences": payload.preferences or ""
        }
        
        # Run evaluation
        eval_results = meal_plan_evaluation.comprehensive_ragas_evaluation(
            meal_plan=meal_plan_for_eval,
            query=query_for_eval,
            candidate_recipes=candidate_recipes_full,
            llm_output=resp,
            include_custom_metrics=True
        )
        
        logging.info(f"Meal plan evaluation completed. Overall score: {eval_results.get('overall_score', 0):.3f} (Grade: {eval_results.get('grade', 'N/A')})")
    except Exception as e:
        logging.warning(f"Meal plan evaluation failed: {str(e)}. Continuing without evaluation.")
    
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
    
@app.post(
    "/generate-shopping-list",
    response_model=GenerateShoppingListResponse,
    tags=["Chat LLM"],
    summary="Generate shopping cart using Cohere chat LLM", 
)
def generate_shopping_list(payload: GenerateShoppingListRequest = Body(...)):
    """Generate shopping cart using Cohere chat LLM."""
    if not COHERE_API_KEY:
        raise HTTPException(status_code=503, detail={"message": "Cohere API key not configured"})
    
    # Ensure ingredients is a list of lists
    ingredients = payload.meal_descriptions
    if not isinstance(ingredients[0], list):
        ingredients = [ingredients]

    user_message = f"""
You are a precise grocery list consolidator.

TASK
- Given a week's recipes (list of lists of ingredient strings), produce a single consolidated shopping list.
- For each ingredient: parse (name, quantity, unit), normalize the name, standardize the unit, convert quantities, aggregate duplicates, ASSIGN A CATEGORY, and **preserve essential qualifiers** so different real items are not merged.

ALLOWED CATEGORIES (choose exactly one per item)
Fruits, Vegetables, Dairy, Bread and Baked Goods, Meat and Fish, Meat Alternatives,
Cans and Jars, Pasta, Rice, and Cereals, Sauces and Condiments, Herbs and Spices,
Frozen Foods, Snacks, Drinks

NAME NORMALIZATION
- Handle aliases & minor spelling/casing/plural variants.
- **Strip non-essential descriptors only** (prep/size/state words) from the name.
- **Preserve essential qualifiers** that distinguish different products (see rules below).
- Examples of aliases (non-exhaustive; apply sensibly):
  scallion/scallions/green onions → green onion
  coriander leaves → cilantro
  garbanzo beans/chick peas → chickpeas
  all purpose flour/ap flour/plain flour → all-purpose flour
  caster sugar/granulated sugar → white sugar
  powdered sugar/icing sugar → confectioners sugar
  bicarb soda/bicarbonate of soda → baking soda
  extra virgin olive oil/EVOO → olive oil
  minced beef/ground beef → beef (ground)
  boneless skinless chicken breast → chicken breast

STRIP WORDS (keep core food): fresh, frozen, dried, unsalted, salted, low-sodium, reduced-sodium, organic,
large, small, medium, finely, chopped, diced, minced, sliced, shredded, grated, thawed, peeled,
seeded, cored, trimmed, packed, raw, cooked, unsweetened, sweetened

UNIT STANDARDIZATION (use only these):
- volume → ml
- mass → g
- count → ea

CONVERSIONS (imperial → metric; do NOT infer densities):
Volume: tsp=4.92892 ml, tbsp=14.7868 ml, cup=236.588 ml, fl oz=29.5735 ml, pint=473.176 ml, quart=946.353 ml, liter/litre=1000 ml, ml=1
Mass: g=1, kg=1000, oz=28.3495 g, lb=453.592 g

COUNTABLE DEFAULTS
If unit missing AND item is countable (egg, onion, green onion, garlic clove, shallot, lime, tomato, avocado, tortilla, etc.) → use ea.
Otherwise assume g and do not invent mass; if you cannot convert, keep the original dimension family (no ml↔g guessing).

AMBIGUITY RULES
- Ranges (e.g., "2–3 cloves") → use the mean (2.5).
- If package size is unknown (e.g., "1 jar capers"), treat as ea and aggregate by ea.
- Normalize plural/singular to a singular concept for the name (e.g., "onions" → "onion") but keep quantities correct.

AGGREGATION (with qualifier protection) 
- Sum quantities for identical (normalized_name, unit) pairs.
- Do NOT return duplicates.
- **Do NOT merge** items that differ by the following (non-exhaustive; apply sensibly) **essential qualifiers**:
  • **Sugars**: keep separate → white sugar (granulated/caster), brown sugar (light/dark), confectioners sugar
  • **Milk & milk alternatives**: keep separate by type and fat % → whole milk, 2% milk, 1% milk, skim milk, evaporated milk, condensed milk, almond milk, soy milk, oat milk
  • **Cheese**: keep separate by variety → cheddar, mozzarella, parmesan, feta, goat cheese, cream cheese, ricotta, etc. (ignore state like shredded/sliced/block)
  • **Flour**: keep separate by type → all-purpose flour, bread flour, cake flour, whole wheat flour, almond flour, etc.
  • **Rice/Grains**: keep separate by type → white rice, brown rice, basmati, jasmine, quinoa, oats
  • **Pasta**: shapes can be separate if explicitly specified (spaghetti, penne, macaroni)
  • **Oils**: keep separate by base → olive oil, canola oil, vegetable oil, avocado oil; (extra-virgin vs light may be merged to "olive oil" unless explicitly required)
  • **Vinegars**: keep separate → white vinegar, apple cider vinegar, balsamic vinegar, rice vinegar
  • **Salts**: keep separate → table salt, kosher salt, sea salt (do not merge across)
  • **Beans/Legumes**: keep separate by species → chickpeas, black beans, kidney beans, lentils
  • **Butter vs Margarine vs Shortening**: do not merge
  • **Yogurts**: keep separate by base/type (greek yogurt vs regular; dairy vs plant)

CATEGORIZATION GUIDANCE (non-exhaustive; apply sensibly)
- Fruits: apple, banana, berries, citrus, avocado, etc.
- Vegetables: onion, garlic, potato, greens, peppers, tomato (default tomato to Vegetables for store navigation).
- Dairy: milk, butter, cheese, yogurt, cream.
- Bread and Baked Goods: bread, buns, tortillas, pitas, pastry.
- Meat and Fish: beef, pork, chicken, turkey, fish, seafood.
- Meat Alternatives: tofu, tempeh, seitan, plant-based meats, legumes used as protein.
- Cans and Jars: canned beans, tomato paste, olives, pickles, capers, jarred sauces.
- Pasta, Rice, and Cereals: pasta/noodles, rice, quinoa, oats, flour.
- Sauces and Condiments: oils, vinegars, ketchup, mustard, mayo, soy sauce, hot sauce.
- Herbs and Spices: cilantro, parsley, basil, dried spices, blends.
- Frozen Foods: anything explicitly frozen.
- Snacks: chips, crackers, cookies, bars, nuts-as-snacks.
- Drinks: water, juice, soda, coffee, tea, milk alternatives as beverages.

    - OUTPUT EXACTLY THIS JSON (no explanation, no markdown):

    {{
    "shopping_list": [
        {{
        "category": "category_name",
        "name": "ingredient_name",
        "unit": "unit_name",
        "quantity": 0.0
        }}
    ]
    }}

    Now, here are the ingredients as JSON:
    {json.dumps(ingredients, indent=2)}
    """

    try:
        message = [{"role": "user", "content": user_message}]
        resp = _cohere_chat(message, model=payload.model or "command-a-03-2025", json_enforce=True)
        
        # Try to parse the response as JSON
        try:
            format_resp = json.loads(resp)
            if not isinstance(format_resp, dict) or "shopping_list" not in format_resp:
                raise ValueError("Response missing required shopping_list field")
            
            print(f"Parsed shopping list response: {format_resp}")
            return GenerateShoppingListResponse(
                shopping_list=format_resp["shopping_list"],
                notes=format_resp.get("notes")
            )
        except json.JSONDecodeError as e:
            # If JSON parsing fails, log the response for debugging
            logging.error(f"Failed to parse response as JSON: {resp}")
            raise HTTPException(
                status_code=500,
                detail={"message": f"Failed to parse LLM response as JSON: {str(e)}"}
            )
        except ValueError as e:
            # If response format is invalid, return error
            logging.error(f"Invalid response format: {resp}")
            raise HTTPException(
                status_code=500,
                detail={"message": f"Invalid response format: {str(e)}"}
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"message": f"LLM call failed: {type(e).__name__}: {str(e)}"}
        )
    

async def _normalize_batch_task(
    co,
    batch: list[str],
    model: str,
    semaphore: asyncio.Semaphore,
    executor: ThreadPoolExecutor,
) -> list[dict]:
    """
    One batch task:
    - builds the prompt
    - calls Cohere chat in a thread
    - parses NDJSON into list[dict]
    """
    loop = asyncio.get_running_loop()

    async with semaphore:
        prompt = create_grocery_prompt(batch)
        # logging.info(f"Calling LLM for batch of size {len(batch)} with prompt:\n{prompt}")

        def _call_cohere_sync() -> list[dict]:
            # Adjust attribute access based on the Cohere SDK version you use
            resp = co(
                [{"role": "user", "content": prompt}],
                model=model,
                json_enforce=False,
                timeout=60,
            )

            # logging.info(f"LLM response: {resp}")

            return parse_ndjson_response(resp)

        return await loop.run_in_executor(executor, _call_cohere_sync)
    
async def build_shopping_list_parallel(
    co,
    ingredients_by_meal: list[list[str]],
    model: str = "command-r",
    batch_size: int = 80,
    max_calls_per_second: float = 2.0,
    max_parallel_workers: int = 4,
) -> dict:
    """
    - co: Cohere client
    - ingredients_by_meal: list of ingredient lists (one per meal)
    - model: Cohere model name
    - batch_size: #ingredient strings per LLM call
    - max_calls_per_second: rate limit
    - max_parallel_workers: how many calls we run in parallel (threadpool size)

    Returns:
      {"shopping_list": [...]}
    """

    flat_ingredients = flatten_ingredients(ingredients_by_meal)
    batches = list(chunk_list(flat_ingredients, batch_size))
    logging.info(f"Total ingredients: {len(flat_ingredients)}, divided into {len(batches)} batches of up to {batch_size} each")

    if not batches:
        return {"shopping_list": []}

    semaphore = asyncio.Semaphore(max_parallel_workers)
    executor = ThreadPoolExecutor(max_workers=max_parallel_workers)

    tasks = []
    interval = 1.0 / max_calls_per_second
    start_time = time.monotonic()

    # Schedule tasks, respecting the max_calls_per_second
    for idx, batch in enumerate(batches):
        # Ensure we don't schedule more than N calls per second
        target_time = start_time + idx * interval
        now = time.monotonic()
        delay = max(0.0, target_time - now)
        await asyncio.sleep(delay)

        task = asyncio.create_task(
            _normalize_batch_task(co, batch, model, semaphore, executor)
        )
        tasks.append(task)

    # Wait for all batches to complete
    all_batch_items = await asyncio.gather(*tasks)
    # logging.info(f"All batch items: {all_batch_items}")

    # Flatten list of lists into one list of item dicts
    all_items = [item for batch_items in all_batch_items for item in batch_items]
    # logging.info(all_items)

    # Aggregate using your existing logic
    shopping_list = aggregate_items(all_items)
    # logging.info(f"Aggregated shopping list: {shopping_list}")

    return {"shopping_list": shopping_list}

@app.post(
    "/generate-shopping-list-logical",
    response_model=GenerateShoppingListResponse,
    tags=["Grocery Generation"],
    summary="Generate shopping cart using Cohere chat LLM and logical conversions", 
)
def generate_shopping_list_logical(payload: GenerateShoppingListRequest = Body(...)):
    """Generate shopping cart using Cohere chat LLM and logical conversions."""
    # Time the entire process and log duration
    start_time = time.time()
    if not COHERE_API_KEY:
        raise HTTPException(status_code=503, detail={"message": "Cohere API key not configured"})
    # logging.info(f"Received payload for logical shopping list generation: {payload}")
    
    # Fist use cohere to determine unit families, categories, normalized names, and provide merge keys
    # Then do the conversions and aggregations locally in code
    try:
        shopping_list = asyncio.run(
            build_shopping_list_parallel(
                _cohere_chat,
                payload.meal_descriptions,
                model=payload.model or "command-a-03-2025",
                batch_size=20,
                max_calls_per_second=5.0,
                max_parallel_workers=10,
            )
        )

        # logging.info(f"Received response from LLM: {shopping_list}")
        logging.info(f"Generated shopping list with {len(shopping_list.get('shopping_list', []))} items")
        logging.info(f"Started with {sum(len(meal) for meal in payload.meal_descriptions)} ingredients across {len(payload.meal_descriptions)} meals")
        # print the number of ingredients that were consolidated (difference between input and output counts)
        logging.info(f"Consolidated {sum(len(meal) for meal in payload.meal_descriptions) - len(shopping_list.get('shopping_list', []))} ingredients into final shopping list")

        end_time = time.time()
        duration = end_time - start_time
        logging.info(f"Shopping list generation completed in {duration:.2f} seconds")
        
        return GenerateShoppingListResponse(
            shopping_list=shopping_list.get("shopping_list", []),
            notes=["Generated using logical normalization and aggregation"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"message": f"LLM call failed: {type(e).__name__}: {str(e)}"}
        )

@app.post(
    "/generate-shopping-list-pre-tagged",
    response_model=GenerateShoppingListResponse,
    tags=["Grocery Generation"],
    summary="Generate shopping cart using pre-tagged ingredients with logical conversions", 
)
def generate_shopping_list_pre_tagged(payload: GenerateShoppingListPreTaggedRequest = Body(...)):
    """Generate shopping cart using pre-tagged ingredients with logical conversions."""
    # Time the entire process and log duration
    start_time = time.time()

    # # Write payload to file for debugging
    # with open("pre_tagged_payload.json", "w") as f:
    #     f.write(payload.json())

    # Make sure that the payload includes "merge_ingredients" field for each ingredient
    try:
        # convert MergeIngredient objects to dicts
        payload.merge_ingredients = [dict(mi) for mi in payload.merge_ingredients]
        shopping_list = aggregate_items(payload.merge_ingredients,True)
        logging.info(f"Generated shopping list with {len(shopping_list)} items from {len(payload.merge_ingredients)} pre-tagged ingredients")
        end_time = time.time()
        duration = end_time - start_time
        logging.info(f"Pre-tagged shopping list generation completed in {duration:.2f} seconds")
        
        return GenerateShoppingListResponse(
            shopping_list=shopping_list,
            notes=["Generated using pre-tagged ingredients with logical normalization and aggregation"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"message": f"Aggregation failed: {type(e).__name__}: {str(e)}"}
        )


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
    # prompt = llm_planning.build_messages(payload.num_days, user_input, include_few_shots=False)
    prompt = llm_planning.build_messages_compact(payload.num_days, user_input)
    if readable_flags and prompt and prompt[0].get("role") == "system":
        prompt[0]["content"] = (
            f"{prompt[0]['content']}\n\nDietary constraints to enforce: "
            + ", ".join(readable_flags)
            + "."
        )

    # Call the LLM
    try:
        # Get current time to track how long it takes to get a response
        start_time = time.time()
        resp = _cohere_chat(prompt, model=payload.model or "command-a-03-2025", json_enforce=True)
        end_time = time.time()
        duration = end_time - start_time
        logging.info(f"LLM call completed in {duration:.2f} seconds")
    except Exception as e:
        raise HTTPException(status_code=500, detail={"message": f"LLM call failed: {type(e).__name__}: {e}"})
    # Basic test for now
    # resp = "{\n  \"plan_meta\": {\n    \"days\": 2,\n    \"notes\": \"Mix of pancakes and oatmeal for breakfast, salads for lunch, and hearty chicken or fish dinners.\"\n  },\n  \"days\": [\n    {\n      \"day\": 1,\n      \"breakfast\": {\n        \"title\": \"Blueberry Pancakes\",\n        \"meal_tag\": \"breakfast\",\n        \"query\": \"fluffy blueberry pancakes breakfast\"\n      },\n      \"lunch\": {\n        \"title\": \"Grilled Chicken Caesar Salad\",\n        \"meal_tag\": \"salad\",\n        \"query\": \"grilled chicken caesar salad lunch\"\n      },\n      \"dinner\": {\n        \"title\": \"Baked Salmon with Roasted Vegetables\",\n        \"meal_tag\": \"main\",\n        \"query\": \"baked salmon roasted vegetables hearty dinner\"\n      }\n    },\n    {\n      \"day\": 2,\n      \"breakfast\": {\n        \"title\": \"Apple Cinnamon Oatmeal\",\n        \"meal_tag\": \"breakfast\",\n        \"query\": \"apple cinnamon oatmeal breakfast\"\n      },\n      \"lunch\": {\n        \"title\": \"Greek Salad with Feta\",\n        \"meal_tag\": \"salad\",\n        \"query\": \"greek salad feta lunch\"\n      },\n      \"dinner\": {\n        \"title\": \"Lemon Herb Roasted Chicken\",\n        \"meal_tag\": \"main\",\n        \"query\": \"lemon herb roasted chicken hearty dinner\"\n      }\n    }\n  ]\n}"

    # Validate the LLM response
    valid, format_resp, errors = llm_planning.parse_and_validate_output(resp, payload.num_days)

    if len(errors) > 0:
        logging.warning(f"LLM response parse/validation errors: {errors}")
    if not valid:
        raise HTTPException(status_code=500, detail={"message": f"LLM response format invalid: {format_resp} with errors: {errors}"})

    # return the formatted response
    return GenerateMealIdeasResponse(
        plan_meta=format_resp.get("meta"),
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
    recipes_per_meal = int(np.ceil(payload.candidate_recipes/len(meal_config.days)/3.0)) # we will trim back down after collecting recipes
    recipes_limit = 10 # trim back down after collecting recipes

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

    # Keep track of recipe IDs to avoid duplicates - we want unique recipes only
    recipe_id_list = []

    # Get current time to track how long it takes to get all recipe searches
    start_time = time.time()
    meal_keys = list(nested_candidates.keys())
    for day_cfg in meal_config.days:
        for meal_key in meal_keys:
            query = day_cfg[meal_key]["query"]
            meal_tag = day_cfg[meal_key]["meal_tag"]
            meal_target = payload_targets[meal_key]
            # logging.info(f"Searching for recipes for day {day_cfg['day']} meal {meal_key} with query '{query}' and meal_tag '{meal_tag}'")

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
                limit=recipes_limit,
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

            search_response = _core_search(request_payload, force_limit=recipes_limit)
            # logging.info(f"  Found {search_response.count} results using dense query '{query}'")
            meal_recipe_count = 0
            for recipe_payload in search_response.results:
                # logging.info(f"recipe_payload keys: {list(recipe_payload.keys())}")
                # logging.info(f"    Candidate recipe: {recipe_payload.get('id', 'ERROR')} {recipe_payload.get('title','')} (meal_type={meal_key}, query='{query}')")
                formatted_recipe = _format_meal_result_llm(recipe_payload,meal_key,query)
                if recipe_payload.get("id") not in recipe_id_list:
                    recipe_id_list.append(recipe_payload.get("id"))
                    meal_recipe_count += 1
                else:
                    continue # skip duplicate recipe
                recipe_list_key = f"{day_cfg['day']}"
                if recipe_list_key not in list(nested_candidates[meal_key].keys()):
                    nested_candidates[meal_key][recipe_list_key] = []
                nested_candidates[meal_key][recipe_list_key].append(formatted_recipe)
                if meal_recipe_count >= recipes_per_meal:
                    break

            queries_list.append(query)
    end_time = time.time()
    duration = end_time - start_time
    logging.info(f"All recipe searches completed in {duration:.2f} seconds. Total unique candidate recipes found: {len(recipe_id_list)}")
    
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