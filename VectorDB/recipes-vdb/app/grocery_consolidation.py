from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Iterable
import json

# --- Canonical unit maps ---

VOLUME_TO_ML = {
    "tsp": 4.92892,
    "teaspoon": 4.92892,
    "teaspoons": 4.92892,
    "tbsp": 14.7868,
    "tablespoon": 14.7868,
    "tablespoons": 14.7868,
    "cup": 236.588,
    "cups": 236.588,
    "fl oz": 29.5735,
    "floz": 29.5735,
    "pint": 473.176,
    "pints": 473.176,
    "quart": 946.353,
    "quarts": 946.353,
    "liter": 1000.0,
    "litre": 1000.0,
    "liters": 1000.0,
    "litres": 1000.0,
    "ml": 1.0,
}

MASS_TO_G = {
    "g": 1.0,
    "gram": 1.0,
    "grams": 1.0,
    "kg": 1000.0,
    "kilogram": 1000.0,
    "kilograms": 1000.0,
    "oz": 28.3495,
    "ounce": 28.3495,
    "ounces": 28.3495,
    "lb": 453.592,
    "lbs": 453.592,
    "pound": 453.592,
    "pounds": 453.592,
}

COUNT_UNITS = {"ea", "each", "count"}



def flatten_ingredients(ingredients_by_meal: list[list[str]]) -> list[str]:
    return [ing for meal in ingredients_by_meal for ing in meal]

def chunk_list(lst: list[str], batch_size: int) -> Iterable[list[str]]:
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

def parse_ndjson_response(text: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            items.append(obj)
        except json.JSONDecodeError:
            # Likely a truncated last line if token limit hit – ignore it.
            continue
    return items


def normalize_unit_text(unit: str | None) -> str | None:
    if unit is None:
        return None
    unit = unit.strip().lower()
    if unit in COUNT_UNITS:
        return "ea"
    # sometimes models output plural forms; keep as-is for lookup above
    return unit


# def convert_to_canonical(quantity: float, unit: str | None, unit_family: str) -> tuple[float, str]:
#     """
#     Convert to canonical units:
#       - volume -> ml
#       - mass -> g
#       - count -> ea
#     Returns (quantity_in_canonical, canonical_unit).
#     """
#     unit = normalize_unit_text(unit)

#     if unit_family == "volume":
#         factor = VOLUME_TO_ML.get(unit, None)
#         if factor is None:
#             # Unknown volume unit: leave as-is (fallback)
#             return quantity, unit or "ml"
#         return quantity * factor, "ml"

#     elif unit_family == "mass":
#         factor = MASS_TO_G.get(unit, None)
#         if factor is None:
#             # Unknown mass unit: leave as-is
#             return quantity, unit or "g"
#         return quantity * factor, "g"

#     elif unit_family == "count":
#         # Treat all countables as "ea"
#         return quantity, "ea"

#     # Fallback: no family, return as-is
#     return quantity, unit or ""

def convert_to_canonical(quantity: float, unit: str | None, unit_family: str) -> tuple[float, str]:
    unit = normalize_unit_text(unit)

    if unit_family == "volume":
        factor = VOLUME_TO_ML.get(unit, None)
        if factor is None:
            return quantity, unit or "ml"
        return quantity * factor, "ml"

    elif unit_family == "mass":
        factor = MASS_TO_G.get(unit, None)
        if factor is None:
            return quantity, unit or "g"
        return quantity * factor, "g"

    elif unit_family == "count":
        return quantity, "ea"

    return quantity, unit or ""


# def aggregate_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#     """
#     items: list of dicts from the LLM:
#         {
#           "n": str,
#           "q": float,
#           "u": str,
#           "f": "volume" | "mass" | "count",
#           "c": str,
#           "m": str,
#         }

#     Returns shopping list:
#         [
#           {
#             "category": "...",
#             "name": "...",
#             "unit": "ml" | "g" | "ea",
#             "quantity": float,
#           },
#           ...
#         ]
#     """
#     # Accumulate quantities per (m, c, category)
#     totals: dict[tuple[str, str, str], float] = defaultdict(float)
#     name_lookup: dict[tuple[str, str, str], str] = {}

#     for item in items:
#         quantity = float(item.get("q", 0) or 0)
#         unit_family = item.get("f", "").lower()
#         unit = item.get("u", None)
#         merge_key = item.get("m") or item.get("n")
#         category = item.get("c", "Other")
#         display_name = item.get("n") or merge_key

#         qty_canon, unit_canon = convert_to_canonical(quantity, unit, unit_family)

#         key = (merge_key, unit_canon, category)
#         totals[key] += qty_canon
#         # store a nice display name once
#         if key not in name_lookup:
#             name_lookup[key] = display_name

#     shopping_list: List[Dict[str, Any]] = []
#     for (merge_key, unit_canon, category), total_qty in totals.items():
#         shopping_list.append(
#             {
#                 "category": category,
#                 "name": name_lookup[(merge_key, unit_canon, category)],
#                 "unit": unit_canon,
#                 "quantity": round(float(total_qty), 2),
#             }
#         )

#     # Optional: sort by category then name
#     shopping_list.sort(key=lambda x: (x["category"], x["name"]))
#     # shopping list needs to be a generic list of dicts
#     shopping_list_generic: List[Dict[str, Any]] = [dict(item) for item in shopping_list]
#     return shopping_list_generic

def aggregate_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    items: list of dicts from the LLM:
        {
          "o": str,   # original_name
          "n": str,   # normalized_name
          "q": float, # quantity
          "u": str,   # raw unit text
          "f": str,   # "volume" | "mass" | "count"
          "c": str,   # category
          "m": str,   # merge_key
          "recipe_id": str,   # optional recipe ID
          "title": str,       # optional recipe name
        }

    Returns shopping list:
        [
          {
            "category": "...",
            "name": "...",
            "unit": "ml" | "g" | "ea",
            "quantity": float,
            "recipe_ids": [str],  # optional list of recipe IDs that use this item
            "recipe_names": [str], # optional list of recipe names that use this item
          },
          ...
        ]
    """

    # print(items)

    # totals keyed by (merge_key, canonical_unit, category)
    totals: dict[tuple[str, str, str], tuple[float, list, list]] = defaultdict(lambda: (0.0, [], []))
    name_lookup: dict[tuple[str, str, str], str] = {}
    # track which canonical units (i.e., which dimensions) appear per merge_key
    units_by_merge: dict[str, set[str]] = defaultdict(set)

    for item in items:
        original_name = item.get("o", "")
        quantity = float(item.get("q", 0) or 0)
        unit_family = (item.get("f") or "").lower()
        raw_unit = item.get("u", None)
        merge_key = item.get("m") or item.get("n")
        normalized_name = item.get("n", "") or merge_key
        category = item.get("c", "Other")
        display_name = item.get("n") or merge_key
        recipe_id = item.get("recipe_id", None)
        recipe_name = item.get("title", None)

        # your existing canonicalization
        qty_canon, unit_canon = convert_to_canonical(quantity, raw_unit, unit_family)

        merge_key = normalized_name.strip().lower()
        key = (merge_key, unit_canon, category)
        # log when we consolidate items
        if key in totals.keys():
            print(f"Consolidating '{original_name}' into '{display_name}' under key {key}. raw_unit='{quantity} {raw_unit}', unit_family='{unit_family}' -> {qty_canon} {unit_canon}")

        totals[key] = (
            totals[key][0] + qty_canon,
            totals[key][1] + ([recipe_id] if recipe_id else []) if recipe_id not in totals[key][1] else totals[key][1],
            totals[key][2] + ([recipe_name] if recipe_name else []) if recipe_name not in totals[key][2] else totals[key][2],
        )
        if key not in name_lookup:
            name_lookup[key] = display_name

        units_by_merge[merge_key].add(unit_canon)

    shopping_list: List[Dict[str, Any]] = []
    print(f"Totals computed: {totals}")

    for (merge_key, unit_canon, category), (total_qty, recipe_ids, recipe_names) in totals.items():
        base_name = name_lookup[(merge_key, unit_canon, category)]
        canonical_units = units_by_merge[merge_key]

        # if multiple unit types exist for this ingredient, add a descriptor
        name = base_name
        if len(canonical_units) > 1:
            if unit_canon == "g":
                suffix = " (by weight)"
            elif unit_canon == "ml":
                suffix = " (by volume)"
            elif unit_canon == "ea":
                suffix = " (by count)"
            else:
                suffix = f" ({unit_canon})"
            name = base_name + suffix

        shopping_list.append(
            {
                "category": category,
                "name": name,
                "unit": unit_canon,
                "quantity": round(float(total_qty), 2),
                "recipe_ids": list(set(recipe_ids)),
                "recipe_names": list(set(recipe_names)),
            }
        )

    shopping_list.sort(key=lambda x: (x["category"], x["name"]))
    return shopping_list


def create_grocery_prompt(flat_ingredients: list[str]) -> str:
    ingredients_json = json.dumps(flat_ingredients, ensure_ascii=False)

    return f"""
You are a precise grocery ingredient normalizer.  
Your output must be STRICT and CONSISTENT across all ingredients.

Input:
- You receive a JSON array of ingredient strings.

====================================
TASK  
For EACH ingredient string, output ONE JSON OBJECT on its own line (NDJSON) with fields:
  - "o": original_name (copy verbatim from input ingredient string)
  - "n": normalized_name  
       • singular  
       • lowercase  
       • remove ONLY non-essential prep words ("chopped", "diced", "fresh", "minced", etc.)  
       • KEEP essential qualifiers that define a unique product  
         (brown sugar, whole milk, balsamic vinegar, olive oil, green bell pepper, soy sauce)  
       • standardize obvious synonyms in the NAME space (e.g., scallion/green onion → green onion)  
       • ALWAYS convert plurals to singular (strawberries → strawberry, onions → onion)

  - "q": numeric quantity (float; use the mean for ranges like “2–3” set to 2.5. Convert fractions to decimals: 1 1/2 is equal to 1.5)

  - "u": unit TEXT, copied from the ingredient  
       • Copy the unit phrase as it appears (e.g., "cups", "cup", "oz", "ounces", "tsp", "tablespoons")  
       • Trim spaces and punctuation, but do NOT convert between units  
       • Do NOT standardize to a fixed list; do NOT infer or change the unit family  
       • If there is clearly no unit word (e.g., "2 onions"), use an empty string "".

  - "f": unit_family — must be one of: "volume", "mass", or "count"  
       • Classify based on the unit provided in "u". If no unit is provided, classify as "count".
       • "count" for units like each, count, or without a unit or items normally bought as discrete items (onion, egg, bell pepper, tomato, avocado, garlic clove, etc.)  
       • "mass" for weight measurements (g, kg, oz, lb, etc.)  
       • "volume" for liquid or volume measurements (tsp, tsp, tbsp, cup, cups, ml, liters, etc.)

  - "c": category (exact match from this list):  
      Fruits, Vegetables, Dairy, Bread and Baked Goods, Meat and Fish,  
      Meat Alternatives, Cans and Jars, Pasta, Rice, and Cereals,  
      Sauces and Condiments, Herbs and Spices, Frozen Foods, Snacks, Drinks

  - "m": merge_key  
       • stable canonical version of the ingredient (how it should be grouped)  
       • examples:  
           all-purpose flour / plain flour → flour  
           whole wheat flour → whole wheat flour  
           bell pepper / green pepper → bell pepper  
           scallion / green onion → green onion  
           garbanzo beans / chickpeas → chickpea  
       • sugars MUST stay distinct: white sugar, brown sugar, powdered sugar  

====================================
STRICT NORMALIZATION RULES (apply always)

1. SINGULARIZATION  
   Always output singular names: strawberries → strawberry, tomatoes → tomato, onions → onion.

2. NO UNIT CONVERSION  
   • Do NOT convert from volume to mass or mass to volume.  
   • Do NOT change units; only classify the family in "f".

3. NO ARRAYS, NO EXTRA TEXT  
   Output one JSON object per line.  
   No commas. No brackets. No prose.

====================================

FORMAT EXAMPLE (FOR STRUCTURE ONLY, NOT CONTENT):
{{"o":"1 cup oat","n":"oat","q":1,"u":"cup","f":"volume","c":"Pasta, Rice, and Cereals","m":"oat"}}
{{"o":"0.5 cups brown sugar","n":"brown sugar","q":0.5,"u":"cups","f":"volume","c":"Bread and Baked Goods","m":"brown sugar"}}

====================================

Now output ONE JSON object per line for EACH ingredient in:
{ingredients_json}
"""


# def create_grocery_prompt_concise_flat(flat_ingredients: list[str]) -> str:
#     ingredients_json = json.dumps(flat_ingredients, ensure_ascii=False)

#     return f"""
# You are a grocery ingredient normalizer.

# Input:
# - You receive a JSON array of ingredient strings.

# Task:
# For EACH ingredient string, output ONE JSON OBJECT on its own line (NDJSON) with fields:
#   - "n": normalized_name (singular, lowercase, only basic prep words (unsalted butter, unsweetened almond milk); keep essential qualifiers)
#   - "q": numeric quantity (float; use average for ranges like "2–3")
#   - "u": normalized unit ("tsp","tbsp","cup","ml","g","kg","oz","lb","ea")
#   - "f": unit_family — "volume", "mass", or "count"
#   - "c": category (one of: Fruits, Vegetables, Dairy, Bread and Baked Goods,
#        Meat and Fish, Meat Alternatives, Cans and Jars, Pasta, Rice, and Cereals,
#        Sauces and Condiments, Herbs and Spices, Frozen Foods, Snacks, Drinks)
#   - "m": merge_key for grouping identical ingredients (like "all-purpose flour", "flour", "plain flour" all map to "flour", but "brown sugar" stays as "brown sugar")

# Important rules:
# - Output **one JSON object per line**, no commas, no brackets, no array.
# - Do NOT combine, omit, or invent ingredients.
# - Keep the same number of output lines as input ingredients.
# - No additional text.

# FORMAT EXAMPLE (format only, NOT an array):
# {{"n":"oat","q":1,"u":"cup","f":"volume","c":"Pasta, Rice, and Cereals","m":"oat"}}
# {{"n":"brown sugar","q":0.5,"u":"cup","f":"volume","c":"Bread and Baked Goods","m":"brown sugar"}}

# Now output one JSON object per line for each ingredient in:
# {ingredients_json}
# """

