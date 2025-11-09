# meal_plan_prompts.py
"""
Utilities to build chat prompts for an LLM meal planner and validate the JSON output.

Features
- Prompt assembly (system/assistant/user + optional few-shots) following role paradigms.
- Output parsing helpers (strip code fences, extract JSON).
- Strict validation against the rules you specified:
  * Exactly n days, each with breakfast, lunch, dinner.
  * meal_tag rules: breakfast="breakfast"; lunch in {"salad","soup","main"}; dinner="main".
  * Titles short, no emojis.
  * Query present and "general" (no URLs; at least 3 tokens).
  * No repeated meal titles across the plan.
  * Day numbering sane.

Usage
-----
from meal_plan_prompts import build_messages, parse_and_validate_output

messages = build_messages(n=5, user_brief="Vegetarian, high protein, quick.")
# -> send 'messages' to your chat completion API

ok, plan, errors = parse_and_validate_output(llm_text_output, n=5)
if not ok:
    # handle / ask model to self-correct using 'errors'
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Tuple, Any, Optional

# ---------------------------
# Prompt content (ROLE: system)
# ---------------------------

SYSTEM_PROMPT = """You are MealPlanJSON, an assistant that turns a human preference brief into a n-day meal plan with three meals per day (breakfast, lunch, dinner). You must return ONLY valid JSON, no prose, no comments, no Markdown.

Rules
- Exactly 3 meals per day: breakfast, lunch, dinner.
- meal_tag:
  - breakfast → "breakfast" (always)
  - lunch → one of "salad", "soup", or "main" (only these)
  - dinner → one of "salad", "soup", or "main" (only these)
- Include a general search string in query for each meal (broad keywords someone could paste into a recipe search—no URLs).
- Align to user dietary notes and vibe (vegetarian, Mediterranean, high-protein, quick, kid-friendly, etc.).
- Vary cuisines/ingredients across the week; avoid repeating the same dish name.
- Keep titles short (≤80 chars). No emojis.
- Units, cuisines, and ingredients should be culturally consistent (don’t mix US/metric randomly).
- If constraints conflict, choose the safest interpretation and proceed.

Output schema (strict)
{
  "plan_meta": {
    "days": 7,
    "notes": "brief echo of intent"
  },
  "days": [
    {
      "day": 1,
      "breakfast": { "title": "...", "meal_tag": "breakfast", "query": "..." },
      "lunch":     { "title": "...", "meal_tag": "salad|soup|main", "query": "..." },
      "dinner":    { "title": "...", "meal_tag": "salad|soup|main", "query": "..." }
    }
  ]
}

Validation
- Reject any lunch or dinner meal_tag not in {salad, soup, main}.
- Enforce exactly days = n in plan_meta.days and number of day objects.
- query must be general (e.g., "high protein vegetarian tofu scramble"), with no brand names or links.

Return only the JSON object.
"""

# ---------------------------
# Compact system prompt (try to speed things up)
# ---------------------------
SYSTEM_PROMPT_COMPACT = """JSON ONLY. Build an n-day meal plan from a single user brief.

Rules
- 3 meals/day: breakfast, lunch, dinner.
- Output keys: meta:{n,notes}, days:[{day,b,l,d}]
- Meal objects: {q, mt}. q = general search string (no URLs), 3–12 tokens, lowercase; mt = meal_tag.
- meal_tag (mt): breakfast→"breakfast"; lunch→"salad"|"soup"|"main"; dinner→"salad"|"soup"|"main".
- Follow diet/exclusions/preferences implied by the brief; keep cuisines/ingredients varied; avoid repeated dish ideas.
- Keep q short and useful for recipe search (include meal word when natural).
- If the brief is vague, choose a safe, popular, diverse mix aligned to any stated diet.
- Focus mostly on main meals for lunch/dinner unless brief suggests otherwise.

Schema
{
  "meta": {"n": 7, "notes": "1-sentence paraphrase of brief"},
  "days": [
    {"day": 1,
     "b": {"q": "...", "mt": "breakfast"},
     "l": {"q": "...", "mt": "salad|soup|main"},
     "d": {"q": "...", "mt": "salad|soup|main"}
    }
  ]
}

Validate
- meta.n == number of day objects.
- q has no URLs/brands; 3–12 tokens.
- mt values are valid per meal.
Return only the JSON object."""


# ---------------------------
# Prompt content (ROLE: assistant)
# ---------------------------

ASSISTANT_PROMPT = """When you receive:
n = <INT>;
user_brief = "<free text from user>"

1) Normalize n to an integer between 1 and 14 (cap at 14 if larger).
2) Create a diversified menu aligned with user_brief.
3) Populate the schema exactly. Do not add fields.
4) plan_meta.notes should be a one-sentence paraphrase of user_brief.
5) Prefer queries that include the relevant meal name + key attributes (diet, cuisine, macro focus, time budget). Examples:
   - "vegetarian protein oatmeal berries 10 minute breakfast"
   - "Mediterranean lentil soup lunch make ahead"
   - "sheet pan salmon broccoli weeknight dinner high protein"
"""

# ---------------------------
# Few-shot examples
# ---------------------------

FEW_SHOT_1_USER = (
    'n = 3\n'
    'user_brief = "Vegetarian, high-protein, quick mornings, hearty lunches, diverse world cuisines, no mushrooms."'
)
FEW_SHOT_1_ASSISTANT = {
    "plan_meta": {
        "days": 3,
        "notes": "Vegetarian, high-protein, quick breakfasts, hearty lunches, diverse, no mushrooms."
    },
    "days": [
        {
            "day": 1,
            "breakfast": {
                "title": "Greek Yogurt Parfait with Berries & Hemp",
                "meal_tag": "breakfast",
                "query": "vegetarian high protein greek yogurt parfait berries hemp seeds breakfast 10 minute"
            },
            "lunch": {
                "title": "Lentil & Feta Chopped Salad",
                "meal_tag": "salad",
                "query": "vegetarian high protein lentil chopped salad feta lemon herb lunch make ahead"
            },
            "dinner": {
                "title": "Tofu Stir-Fry with Broccoli & Cashews",
                "meal_tag": "main",
                "query": "vegetarian high protein tofu stir fry broccoli cashew garlic ginger weeknight dinner"
            }
        },
        {
            "day": 2,
            "breakfast": {
                "title": "Peanut Butter Banana Overnight Oats",
                "meal_tag": "breakfast",
                "query": "overnight oats peanut butter banana chia vegetarian high protein breakfast prep"
            },
            "lunch": {
                "title": "Moroccan Chickpea & Tomato Soup",
                "meal_tag": "soup",
                "query": "vegetarian chickpea tomato soup moroccan spices cumin paprika hearty lunch"
            },
            "dinner": {
                "title": "Paneer Tikka Bowls with Brown Rice",
                "meal_tag": "main",
                "query": "vegetarian paneer tikka bowls high protein brown rice sheet pan dinner"
            }
        },
        {
            "day": 3,
            "breakfast": {
                "title": "Egg White Veggie Wrap with Spinach & Avocado",
                "meal_tag": "breakfast",
                "query": "egg white breakfast wrap spinach avocado vegetarian high protein 10 minute"
            },
            "lunch": {
                "title": "Caprese-Style Farro Salad (No Mushrooms)",
                "meal_tag": "salad",
                "query": "vegetarian farro salad caprese tomato basil mozzarella balsamic lunch"
            },
            "dinner": {
                "title": "Thai Basil Tofu with Green Beans",
                "meal_tag": "main",
                "query": "vegetarian thai basil tofu green beans high protein quick skillet dinner"
            }
        }
    ]
}

FEW_SHOT_2_USER = (
    'n = 2\n'
    'user_brief = "Family-friendly, Mediterranean-leaning, light breakfasts, soup or salad for lunch, fish or chicken for dinners; limit dairy."'
)
FEW_SHOT_2_ASSISTANT = {
    "plan_meta": {
        "days": 2,
        "notes": "Family-friendly, Mediterranean-leaning; light breakfasts; soup/salad lunches; fish or chicken dinners; lower dairy."
    },
    "days": [
        {
            "day": 1,
            "breakfast": {
                "title": "Almond Butter Apple Toast (Whole Grain)",
                "meal_tag": "breakfast",
                "query": "light breakfast whole grain toast almond butter apple cinnamon mediterranean inspired"
            },
            "lunch": {
                "title": "Tuscan White Bean & Kale Soup",
                "meal_tag": "soup",
                "query": "mediterranean tuscan white bean kale soup dairy free lunch"
            },
            "dinner": {
                "title": "Lemon Herb Baked Salmon with Roasted Vegetables",
                "meal_tag": "main",
                "query": "mediterranean lemon herb baked salmon sheet pan roasted vegetables dairy free dinner"
            }
        },
        {
            "day": 2,
            "breakfast": {
                "title": "Citrus Yogurt Bowl with Pistachios (Light Dairy)",
                "meal_tag": "breakfast",
                "query": "light breakfast mediterranean citrus yogurt pistachios honey minimal dairy"
            },
            "lunch": {
                "title": "Greek Chopped Salad with Chickpeas",
                "meal_tag": "salad",
                "query": "mediterranean greek chopped salad chickpeas cucumber tomato olive oregano lunch"
            },
            "dinner": {
                "title": "Garlic Oregano Chicken Thighs with Quinoa Tabbouleh",
                "meal_tag": "main",
                "query": "mediterranean garlic oregano chicken thighs quinoa tabbouleh weeknight dinner"
            }
        }
    ]
}

FEW_SHOT_3_USER = (
    'n = 4\n'
    'user_brief = "Budget-minded, 20–30 min dinners, hearty lunches for leftovers, gluten-free, dislikes cilantro."'
)
FEW_SHOT_3_ASSISTANT = {
    "plan_meta": {
        "days": 4,
        "notes": "Budget-minded, 20–30 min dinners, hearty leftover-friendly lunches, gluten-free, no cilantro."
    },
    "days": [
        {
            "day": 1,
            "breakfast": {
                "title": "Berry Chia Pudding",
                "meal_tag": "breakfast",
                "query": "gluten free berry chia pudding make ahead budget breakfast"
            },
            "lunch": {
                "title": "Roasted Sweet Potato & Black Bean Bowl (No Cilantro)",
                "meal_tag": "main",
                "query": "gluten free roasted sweet potato black bean bowl lime yogurt sauce no cilantro lunch"
            },
            "dinner": {
                "title": "One-Pan Lemon Garlic Chicken & Rice",
                "meal_tag": "main",
                "query": "gluten free one pan lemon garlic chicken rice 30 minute budget dinner"
            }
        },
        {
            "day": 2,
            "breakfast": {
                "title": "Scrambled Eggs with Spinach & Feta",
                "meal_tag": "breakfast",
                "query": "gluten free scrambled eggs breakfast spinach feta 10 minute"
            },
            "lunch": {
                "title": "Hearty Vegetable Lentil Soup",
                "meal_tag": "soup",
                "query": "gluten free vegetable lentil soup hearty leftovers lunch"
            },
            "dinner": {
                "title": "Shrimp Fried Rice (Cauliflower or Leftover Rice)",
                "meal_tag": "main",
                "query": "gluten free shrimp fried rice 20 minute skillet dinner budget"
            }
        },
        {
            "day": 3,
            "breakfast": {
                "title": "Cottage Cheese & Pineapple Bowl",
                "meal_tag": "breakfast",
                "query": "gluten free cottage cheese pineapple breakfast high protein"
            },
            "lunch": {
                "title": "Quinoa Greek-Style Salad (No Cilantro)",
                "meal_tag": "salad",
                "query": "gluten free quinoa greek salad tomato cucumber olive no cilantro lunch meal prep"
            },
            "dinner": {
                "title": "Turkey Taco Skillet with Corn & Peppers (GF)",
                "meal_tag": "main",
                "query": "gluten free turkey taco skillet corn peppers 25 minute dinner"
            }
        },
        {
            "day": 4,
            "breakfast": {
                "title": "Peanut Butter Oat Smoothie",
                "meal_tag": "breakfast",
                "query": "gluten free peanut butter oat smoothie banana breakfast budget"
            },
            "lunch": {
                "title": "Roasted Tomato Basil Soup (No Cilantro)",
                "meal_tag": "soup",
                "query": "gluten free roasted tomato basil soup creamy dairy optional no cilantro lunch"
            },
            "dinner": {
                "title": "Garlic Butter Tilapia with Steamed Green Beans",
                "meal_tag": "main",
                "query": "gluten free garlic butter tilapia 20 minute dinner skillet"
            }
        }
    ]
}

# ---------------------------
# Public API: build messages
# ---------------------------

def clamp_days(n: int) -> int:
    """Clamp the requested day count to [1, 14]."""
    try:
        n_int = int(n)
    except Exception:
        n_int = 7
    return max(1, min(14, n_int))

def build_user_prompt(n: int, user_brief: str) -> str:
    """Render the user template that the model expects."""
    n_clamped = clamp_days(n)
    return (
        f"n = {n_clamped}\n"
        f'user_brief = "{user_brief.strip()}"'
    )

def build_user_prompt_compact(n: int, user_brief: str) -> str:
    n_clamped = clamp_days(n)
    # short, machine-friendly, no boilerplate
    return json.dumps({"n": n_clamped, "brief": user_brief.strip()})

def build_messages_compact(n: int, user_brief: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT_COMPACT},
        {"role": "user", "content": build_user_prompt_compact(n, user_brief)},
    ]

def validate_plan_compact(plan: dict, n_expected: int) -> list[str]:
    errs = []
    n_expected = clamp_days(n_expected)

    if not isinstance(plan, dict):
        return ["Top-level must be an object"]

    meta = plan.get("meta")
    days = plan.get("days")
    if not isinstance(meta, dict): errs.append('Missing "meta"')
    if not isinstance(days, list): errs.append('Missing "days"')
    if errs: return errs

    # meta
    if not isinstance(meta.get("n"), int): errs.append('"meta.n" must be int')
    elif meta["n"] != n_expected: errs.append(f'meta.n must equal {n_expected}')
    if not isinstance(meta.get("notes"), str) or not meta["notes"].strip():
        errs.append('"meta.notes" required')

    # days
    if len(days) != n_expected:
        errs.append(f'"days" must have {n_expected} items (found {len(days)})')

    valid_l = {"salad","soup","main"}
    for i, day in enumerate(days, 1):
        if not isinstance(day, dict): errs.append(f"days[{i}] must be object"); continue
        # Make sure day.get("d") can cast to int
        try:
            day_d = int(day.get("day"))
        except Exception:
            day_d = None
        if day_d != i: errs.append(f'days[{i}].day must equal {i}')

        for meal_key, mt_req in (("b","breakfast"), ("l",valid_l), ("d",valid_l)):
            m = day.get(meal_key)
            if not isinstance(m, dict):
                errs.append(f'days[{i}].{meal_key} must be object'); continue
            q, mt = m.get("q"), m.get("mt")
            if not isinstance(q, str) or len(q.split()) > 12:
                errs.append(f'days[{i}].{meal_key}.q must be <=12 words. Found {len(q.split())}')
            if _contains_url(q or ""):
                errs.append(f'days[{i}].{meal_key}.q must not contain URLs')
            if meal_key == "l":
                if mt not in valid_l: errs.append(f'days[{i}].l.mt invalid: {mt}')
            elif meal_key == "d":
                if mt not in valid_l: errs.append(f'days[{i}].d.mt invalid: {mt}')
            else:
                if mt != ("breakfast" if meal_key=="b" else "main"):
                    errs.append(f'days[{i}].{meal_key}.mt invalid: {mt}')
    return errs

def reformat_plan_compact(plan: dict) -> dict:
    """Convert compact plan format to full format."""
    full_plan = {
        "plan_meta": {
            "days": plan["meta"]["n"],
            "notes": plan["meta"]["notes"]
        },
        "days": []
    }
    for day in plan["days"]:
        full_day = {
            "day": day["day"],
            "breakfast": {
                "title": "",
                "meal_tag": "breakfast",
                "query": day["b"]["q"]
            },
            "lunch": {
                "title": "",
                "meal_tag": day["l"]["mt"],
                "query": day["l"]["q"]
            },
            "dinner": {
                "title": "",
                "meal_tag": day["d"]["mt"],
                "query": day["d"]["q"]
            }
        }
        full_plan["days"].append(full_day)
    return full_plan


def build_messages(n: int, user_brief: str, include_few_shots: bool = True) -> List[Dict[str, str]]:
    """
    Build a role-based message list for Chat Completions-style APIs.

    Returns a list like:
    [
      {"role": "system", "content": SYSTEM_PROMPT},
      {"role": "assistant", "content": ASSISTANT_PROMPT},
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "<json for fs1>"},
      {"role": "user", "content": "<fs1 user>"},
      ...
    ]
    Few-shots are ordered as (user -> assistant) pairs before the final user prompt
    for better grounding.
    """
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "assistant", "content": ASSISTANT_PROMPT},
    ]

    if include_few_shots:
        # Few-shot #1
        messages.append({"role": "user", "content": FEW_SHOT_1_USER})
        messages.append({"role": "assistant", "content": json.dumps(FEW_SHOT_1_ASSISTANT)})
        # Few-shot #2
        messages.append({"role": "user", "content": FEW_SHOT_2_USER})
        messages.append({"role": "assistant", "content": json.dumps(FEW_SHOT_2_ASSISTANT)})
        # Few-shot #3
        messages.append({"role": "user", "content": FEW_SHOT_3_USER})
        messages.append({"role": "assistant", "content": json.dumps(FEW_SHOT_3_ASSISTANT)})

    # Final user message with the real inputs
    messages.append({"role": "user", "content": build_user_prompt(n, user_brief)})

    return messages

# ---------------------------
# Output parsing & validation
# ---------------------------

_URL_RE = re.compile(r"https?://|www\.", re.I)
# Basic emoji range (covers common emoji blocks)
_EMOJI_RE = re.compile(
    "["                           # start char class
    "\U0001F300-\U0001F5FF"       # symbols & pictographs
    "\U0001F600-\U0001F64F"       # emoticons
    "\U0001F680-\U0001F6FF"       # transport & map
    "\U0001F700-\U0001F77F"       # alchemical
    "\U0001F780-\U0001F7FF"       # geometric
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002702-\U000027B0"       # dingbats
    "\U000024C2-\U0001F251"
    "]",
    flags=re.UNICODE
)

def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences if present."""
    text = text.strip()
    if text.startswith("```"):
        # remove starting triple backticks and optional language tag
        text = re.sub(r"^```[a-zA-Z0-9]*\n?", "", text)
        # remove trailing ```
        text = re.sub(r"\n?```$", "", text)
    return text.strip()

def _extract_json(text: str) -> str:
    """
    Heuristically extract the first top-level JSON object from text.
    Looks for the first '{' and last '}'.
    """
    text = _strip_code_fences(text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object detected in model output.")
    return text[start : end + 1]

def parse_model_output_to_dict(text: str) -> Dict[str, Any]:
    """
    Extract and parse JSON from the LLM text output.
    Raises ValueError on failure.
    """
    json_str = _extract_json(text)
    return json.loads(json_str)

def _is_nonempty_string(x: Any) -> bool:
    return isinstance(x, str) and x.strip() != ""

def _contains_url(s: str) -> bool:
    return bool(_URL_RE.search(s))

def _contains_emoji(s: str) -> bool:
    return bool(_EMOJI_RE.search(s))

def _token_count(s: str) -> int:
    return len([t for t in s.strip().split() if t])

def _path_exists(d: Dict[str, Any], path: List[str], typ: type) -> bool:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return False
        cur = cur[key]
    return isinstance(cur, typ)

def validate_plan(plan: Dict[str, Any], n_expected: int) -> List[str]:
    """
    Validate the meal plan dictionary against all rules.
    Returns a list of human-readable error strings. Empty list means valid.
    """
    errors: List[str] = []
    n_expected = clamp_days(n_expected)

    # Top-level keys
    if not isinstance(plan, dict):
        return ["Top-level output must be a JSON object."]
    if "plan_meta" not in plan or "days" not in plan:
        errors.append('Missing required top-level keys: "plan_meta" and/or "days".')

    # plan_meta
    if _path_exists(plan, ["plan_meta", "days"], int):
        if plan["plan_meta"]["days"] != n_expected:
            errors.append(f'plan_meta.days must equal {n_expected}.')
    else:
        errors.append("plan_meta.days (int) is required.")

    if not _path_exists(plan, ["plan_meta", "notes"], str):
        errors.append("plan_meta.notes (str) is required.")

    # days array
    days = plan.get("days")
    if not isinstance(days, list):
        errors.append('"days" must be an array.')
        return errors

    if len(days) != n_expected:
        errors.append(f'"days" array must have exactly {n_expected} elements (found {len(days)}).')

    seen_titles: set[str] = set()
    seen_day_numbers: set[int] = set()

    for i, day in enumerate(days, start=1):
        if not isinstance(day, dict):
            errors.append(f"days[{i}] must be an object.")
            continue

        # day index
        day_num = day.get("day")
        if not isinstance(day_num, int):
            errors.append(f'days[{i}].day must be an integer (found {type(day_num)}).')
        else:
            if day_num in seen_day_numbers:
                errors.append(f"Duplicate day number {day_num}.")
            seen_day_numbers.add(day_num)

        # required meal keys
        for meal in ("breakfast", "lunch", "dinner"):
            if meal not in day or not isinstance(day[meal], dict):
                errors.append(f'days[{i}].{meal} must be an object.')
                continue

            m = day[meal]
            title = m.get("title")
            meal_tag = m.get("meal_tag")
            query = m.get("query")

            if not _is_nonempty_string(title):
                errors.append(f'days[{i}].{meal}.title must be a non-empty string.')
            else:
                if len(title) > 80:
                    errors.append(f'days[{i}].{meal}.title exceeds 80 characters.')
                if _contains_emoji(title):
                    errors.append(f'days[{i}].{meal}.title must not contain emojis.')
                # track duplicates across the plan
                norm_title = title.strip().lower()
                if norm_title in seen_titles:
                    errors.append(f'Duplicate meal title detected: "{title}" on day {day_num}.')
                seen_titles.add(norm_title)

            if not _is_nonempty_string(meal_tag):
                errors.append(f'days[{i}].{meal}.meal_tag must be a non-empty string.')
            else:
                if meal == "breakfast" and meal_tag != "breakfast":
                    errors.append(f'days[{i}].breakfast.meal_tag must equal "breakfast" (found "{meal_tag}").')
                if meal == "lunch" and meal_tag not in {"salad", "soup", "main"}:
                    errors.append(f'days[{i}].lunch.meal_tag must be one of "salad", "soup", "main" (found "{meal_tag}").')
                if meal == "dinner" and meal_tag not in {"salad", "soup", "main"}:
                    errors.append(f'days[{i}].dinner.meal_tag must be one of "salad", "soup", "main" (found "{meal_tag}").')

            if not _is_nonempty_string(query):
                errors.append(f'days[{i}].{meal}.query must be a non-empty string.')
            else:
                if _contains_url(query):
                    errors.append(f'days[{i}].{meal}.query must be a general search string (no URLs).')
                if _token_count(query) < 3:
                    errors.append(f'days[{i}].{meal}.query looks too short to be general (need ≥3 tokens).')

    # Optional: enforce day numbers are 1..n (not strictly required but helpful)
    if seen_day_numbers and (min(seen_day_numbers) != 1 or max(seen_day_numbers) != n_expected or len(seen_day_numbers) != n_expected):
        errors.append(f'Day numbers should be unique and span 1..{n_expected}.')

    return errors

def parse_and_validate_output(raw_text: str, n_expected: int) -> Tuple[bool, Optional[Dict[str, Any]], List[str]]:
    """
    Parse the model's raw text output into JSON and run validation.
    Returns (ok, plan_dict_or_none, errors).
    """
    try:
        plan = parse_model_output_to_dict(raw_text)
    except Exception as e:
        return False, None, [f"Failed to parse JSON from model output: {e}"]

    # errors = validate_plan(plan, n_expected)
    errors = validate_plan_compact(plan, n_expected)
    return (len(errors) == 0, reformat_plan_compact(plan) if not errors else None, errors)

# ---------------------------
# (Optional) Self-test
# ---------------------------

if __name__ == "__main__":
    # Quick sanity test using a few-shot as surrogate output
    msg = build_messages(3, "Vegetarian, high-protein, quick mornings, hearty lunches, no mushrooms.")
    print(f"[debug] built {len(msg)} messages")

    ok, plan, errs = parse_and_validate_output(json.dumps(FEW_SHOT_1_ASSISTANT), n_expected=3)
    print("Validation OK:", ok)
    if not ok:
        print("\n".join(errs))
    else:
        print("Plan days:", plan["plan_meta"]["days"])
