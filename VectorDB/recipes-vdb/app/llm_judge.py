from __future__ import annotations

import json
import re
from typing import Dict, List, Tuple, Any, Optional
from llm_planning import parse_model_output_to_dict


def generate_system_prompt(daily_cal, daily_protein, daily_fat, daily_carbs, num_days,dietary, preferences, exclusions, candidates, question="Create a balanced 7-day meal plan"):
    prompt = f"""
You are a meal planner AI assistant. Select 7 days of meals (breakfast, lunch, dinner) close to daily macro goals.

USER PREFERENCES: {preferences}
EXCLUSIONS: {exclusions}
DIETARY FLAGS: {dietary}
DAILY MACRO TARGETS: {{"cal":{daily_cal}, "P": {daily_protein}, "F": {daily_fat}, "C": {daily_carbs}}}

FIELDS: [recipe_id, meal, title, cal, P, F, C, ingredients]

CANDIDATES: [
  {recipes_to_list(candidates)}
]

Please choose {num_days*3} recipes ({num_days} per meal type) balancing macros, variety, and exclusions.
return ONLY:
{{
  "day_1": {{"breakfast": "<recipe_id>", "lunch": "<recipe_id>", "dinner": "<recipe_id>"}},
  "day_2": {{"breakfast": "<recipe_id>", "lunch": "<recipe_id>", "dinner": "<recipe_id>"}},
  "day_3": {{"breakfast": "<recipe_id>", "lunch": "<recipe_id>", "dinner": "<recipe_id>"}},
  "day_4": {{"breakfast": "<recipe_id>", "lunch": "<recipe_id>", "dinner": "<recipe_id>"}},
  "day_5": {{"breakfast": "<recipe_id>", "lunch": "<recipe_id>", "dinner": "<recipe_id>"}},
  "day_6": {{"breakfast": "<recipe_id>", "lunch": "<recipe_id>", "dinner": "<recipe_id>"}},
  "day_7": {{"breakfast": "<recipe_id>", "lunch": "<recipe_id>", "dinner": "<recipe_id>"}}
}}

Do NOT include any additional text or commentary.
"""
    return [
        {"role": "user", "content": prompt}
    ]

def recipes_to_list(candidates):
    candidates = json.loads(candidates)
    recipe_list = []
    for recipe in candidates:
        # print(recipe)
        recipe_str = f"[{recipe['recipe_id']}, {recipe['meal_type']}, {recipe['title']}, {recipe['calories']}, {recipe['protein_g']}, {recipe['fat_g']}, {recipe['carbs_g']}, {'. '.join([r for r in recipe['ingredients']])}]"
        recipe_list.append(recipe_str)
    return "\n  ".join(recipe_list)

# def generate_system_prompt(daily_cal, daily_protein, daily_fat, daily_carbs, num_days,dietary, preferences, exclusions, candidates, question="Create a balanced 7-day meal plan"):
#     prompt = f"""
# You are a meal planner AI assistant. Your job is to choose recipes for a 7-day plan based on calorie and macro goals.

# USER PREFERENCES:
# {preferences}

# EXCLUSIONS:
# {exclusions}

# DIETARY FLAGS:
# {dietary}

# DAILY MACRO TARGETS:
# Calories: {daily_cal}
# Protein: {daily_protein}g
# Fat: {daily_fat}g
# Carbs: {daily_carbs}g

# Below are candidate recipes with IDs and macros:

# {candidates}

# ---

# Please choose a combination of recipes for a {num_days}-day plan:
# - 1 breakfast, 1 lunch, and 1 dinner per day (21 meals total)
# - Try to balance the total macros close to the daily targets.
# - Respect user preferences and exclusions.
# - Respond ONLY with a JSON list in this format:

# {{
#   "day_1": {{"breakfast": "<recipe_id>", "lunch": "<recipe_id>", "dinner": "<recipe_id>"}},
#   "day_2": {{"breakfast": "<recipe_id>", "lunch": "<recipe_id>", "dinner": "<recipe_id>"}},
#   "day_3": {{"breakfast": "<recipe_id>", "lunch": "<recipe_id>", "dinner": "<recipe_id>"}},
#   "day_4": {{"breakfast": "<recipe_id>", "lunch": "<recipe_id>", "dinner": "<recipe_id>"}},
#   "day_5": {{"breakfast": "<recipe_id>", "lunch": "<recipe_id>", "dinner": "<recipe_id>"}},
#   "day_6": {{"breakfast": "<recipe_id>", "lunch": "<recipe_id>", "dinner": "<recipe_id>"}},
#   "day_7": {{"breakfast": "<recipe_id>", "lunch": "<recipe_id>", "dinner": "<recipe_id>"}}
# }}

# Do NOT include any additional text or commentary.
# """
#     return [
#         {"role": "user", "content": prompt}
#     ]

# def generate_system_prompt(daily_cal, daily_protein, daily_fat, daily_carbs, num_days,dietary, preferences, exclusions, candidates, question="Create a balanced 7-day meal plan"):
#     prompt = f"""You are a nutrition assistant helping to build a {num_days}-day meal plan.
# Each day should have 3 meals: Breakfast, Lunch, and Dinner.

# Your goal is to select meals so that the **daily totals** are as close as possible to the targets below:

# - Calories: {daily_cal} kcal
# - Protein: {daily_protein} g
# - Fat: {daily_fat} g
# - Carbs: {daily_carbs} g

# User preferences to consider (e.g. meals that the user would prefer, cuisine types, flavors):
# {preferences}

# User exclusions to consider (e.g. meals and ingredients the user want to avoid, dislikes, allergies)
# {exclusions}

# User dietary restrictions to consider(e.g. vegetarian, pescatarian, gluten-free, dairy-free, vegan)
# {dietary}

# Here is a list of candidate meals with macros, flavors, and instructions:
# ---
# {candidates}
# ---

# Question:
# {question}

# Return the meal plan strictly as valid JSON using this structure including the recipe_id values of the relevant recipes from the candidate meals. Do not make up recipe ids. Only use ids listed in the recipe_id field in the candidates:
# {{
#   "days": [
#     {{
#       "day": 1,
#       "meals": {{
#         "breakfast": {{
#           "title": "<Recipe Title>",
#           "calories": 0,
#           "protein_g": 0,
#           "fat_g": 0,
#           "carbs_g": 0,
#           "recipe_id": "<recipe_id>"
#         }},
#         "lunch": {{
#           "title": "<Recipe Title>",
#           "calories": 0,
#           "protein_g": 0,
#           "fat_g": 0,
#           "carbs_g": 0,
#           "recipe_id": "<recipe_id>"
#         }},
#         "dinner": {{
#           "title": "<Recipe Title>",
#           "calories": 0,
#           "protein_g": 0,
#           "fat_g": 0,
#           "carbs_g": 0,
#           "recipe_id": "<recipe_id>"
#         }}
#       }}
#     }}
#   ]
# }}
# """
#     return [
#         {"role": "user", "content": prompt}
#     ]

# -----------------------------
# Output parsing and validation
# -----------------------------
def extract_meal_plan_json(llm_response):
    '''
    Return a structured meal plan JSON object.
    '''

    # Format the LLM response
    return parse_model_output_to_dict(llm_response)