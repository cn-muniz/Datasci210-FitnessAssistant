from __future__ import annotations

import json
import re
from typing import Dict, List, Tuple, Any, Optional
from llm_planning import parse_model_output_to_dict

def generate_system_prompt(daily_cal, daily_protein, daily_fat, daily_carbs, num_days,dietary, preferences, exclusions, candidates, question="Create a balanced 7-day meal plan"):
    prompt = f"""You are a nutrition assistant helping to build a {num_days}-day meal plan.
Each day should have 3 meals: Breakfast, Lunch, and Dinner.

Your goal is to select meals so that the **daily totals** are as close as possible to the targets below:

- Calories: {daily_cal} kcal
- Protein: {daily_protein} g
- Fat: {daily_fat} g
- Carbs: {daily_carbs} g

User preferences to consider (e.g. meals that the user would prefer, cuisine types, flavors):
{preferences}

User exclusions to consider (e.g. meals and ingredients the user want to avoid, dislikes, allergies)
{exclusions}

User dietary restrictions to consider(e.g. vegetarian, pescatarian, gluten-free, dairy-free, vegan)
{dietary}

Here is a list of candidate meals with macros, flavors, and instructions:
---
{candidates}
---

Question:
{question}

Return the meal plan strictly as valid JSON using this structure including the recipe_id values of the relevant recipes from the candidate meals. Do not make up recipe ids. Only use ids listed in the recipe_id field in the candidates:
{{
  "days": [
    {{
      "day": 1,
      "meals": {{
        "breakfast": {{
          "title": "<Recipe Title>",
          "calories": 0,
          "protein_g": 0,
          "fat_g": 0,
          "carbs_g": 0,
          "recipe_id": "<recipe_id>"
        }},
        "lunch": {{
          "title": "<Recipe Title>",
          "calories": 0,
          "protein_g": 0,
          "fat_g": 0,
          "carbs_g": 0,
          "recipe_id": "<recipe_id>"
        }},
        "dinner": {{
          "title": "<Recipe Title>",
          "calories": 0,
          "protein_g": 0,
          "fat_g": 0,
          "carbs_g": 0,
          "recipe_id": "<recipe_id>"
        }}
      }}
    }}
  ]
}}
"""
    return [
        {"role": "user", "content": prompt}
    ]

# -----------------------------
# Output parsing and validation
# -----------------------------
def extract_meal_plan_json(llm_response):
    '''
    Return a structured meal plan JSON object.
    '''

    # Format the LLM response
    return parse_model_output_to_dict(llm_response)