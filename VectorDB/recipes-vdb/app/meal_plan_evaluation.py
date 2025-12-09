"""
RAGAS evaluation for meal planning results.
Evaluates the quality of LLM-generated meal plans.
"""

import logging
import math
import os

logger = logging.getLogger(__name__)

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    ContextRelevance
)
from datasets import Dataset
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper



def prepare_ragas_dataset(meal_plan, query, candidate_recipes, llm_output):
    """
    Convert your meal plan into RAGAS-compatible format
    
    RAGAS expects:
    - question: User's query
    - answer: LLM's generated response
    - contexts: Retrieved documents (candidate recipes)
    - ground_truth: Ideal answer (optional, but improves metrics)
    """
    
    # Format the user query - use values from query, no hardcoded defaults
    num_days = query.get('num_days', 1)
    target_calories = query.get('target_calories', 0)
    target_protein = query.get('target_protein', 0)
    target_fat = query.get('target_fat', 0)
    target_carbs = query.get('target_carbs', 0)
    exclusions = query.get('exclusions', [])
    dietary = query.get('dietary', [])
    
    question = f"""Create a {num_days}-day meal plan with:
    - Daily calories: {target_calories} kcal
    - Protein: {target_protein}g
    - Fat: {target_fat}g  
    - Carbs: {target_carbs}g
    - Exclusions: {exclusions}
    - Dietary: {dietary}
    """
    
    # Format the LLM's answer (the meal plan)
    answer = format_meal_plan_as_text(meal_plan)
    
    # Format contexts (candidate recipes)
    contexts = []
    for recipe in candidate_recipes:
        if recipe is not None:
            if hasattr(recipe, 'dict'):
                recipe = recipe.dict()
            contexts.append(format_recipe_as_context(recipe))
    
    if not contexts:
        contexts = [""]
    
    data_dict = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts], 
    }
    
    logger.info(f"RAGAS dataset prepared: question length={len(question)}, answer length={len(answer)}, contexts count={len(contexts)}")
    
    return data_dict


def _extract_macro_value(macro_str):
    """Extract numeric value from macro string like '150g' -> 150.0"""
    if macro_str is None:
        return 0.0
    if isinstance(macro_str, (int, float)):
        return float(macro_str)
    if isinstance(macro_str, str):
        cleaned = macro_str.replace('g', '').replace('G', '').strip()
        try:
            return float(cleaned)
        except (ValueError, TypeError):
            return 0.0
    return 0.0


def format_meal_plan_as_text(meal_plan):
    """Convert meal plan to text format for RAGAS"""
    lines = []
    
    days = meal_plan.get("days", [])
    for day_data in days:
        if hasattr(day_data, 'dict'):
            day_data = day_data.dict()
        
        day_num = day_data.get("day", 1)
        lines.append(f"Day {day_num}:")
        
        meals = day_data.get("meals", [])
        for meal in meals:
            if meal is None:
                continue
            
            if hasattr(meal, 'dict'):
                meal = meal.dict()
                
            meal_type = meal.get("meal_type", "unknown")
            title = meal.get('title', 'N/A')
            recipe_id = meal.get('recipe_id') or meal.get('id', 'N/A')
            calories = meal.get('calories', 0) or 0

            macros = meal.get('macros', {})
            if isinstance(macros, dict) and macros:
                protein = _extract_macro_value(macros.get('protein', '0g'))
                fat = _extract_macro_value(macros.get('fat', '0g'))
                carbs = _extract_macro_value(macros.get('carbs', '0g'))
            else:
                protein = meal.get('protein_g', 0) or 0
                fat = meal.get('fat_g', 0) or 0
                carbs = meal.get('carbs_g', 0) or 0
            
            lines.append(
                f"  {meal_type.title()}: {title} "
                f"(ID: {recipe_id}, "
                f"{calories} kcal, "
                f"P:{protein}g F:{fat}g C:{carbs}g)"
            )
        
        total_calories = day_data.get('total_calories', 0) or 0
        total_protein = day_data.get('total_protein', 0) or 0
        total_fat = day_data.get('total_fat', 0) or 0
        total_carbs = day_data.get('total_carbs', 0) or 0
        
        lines.append(
            f"  Daily Total: {total_calories} kcal, "
            f"P:{total_protein}g F:{total_fat}g C:{total_carbs}g"
        )
        lines.append("")
    
    return "\n".join(lines)


def format_recipe_as_context(recipe):
    """Format a single recipe as context string"""
    if recipe is None:
        return ""
    
    if hasattr(recipe, 'dict'):
        recipe = recipe.dict()
    
    recipe_id = recipe.get('recipe_id') or recipe.get('id', 'N/A')
    title = recipe.get('title', 'N/A')
    calories = recipe.get('calories', 0) or 0
    protein = recipe.get('protein_g', 0) or 0
    fat = recipe.get('fat_g', 0) or 0
    carbs = recipe.get('carbs_g', 0) or 0
    tags = recipe.get('tags', []) or []
    description = recipe.get('description') or recipe.get('summary', '')
    
    return (
        f"Recipe ID: {recipe_id}\n"
        f"Title: {title}\n"
        f"Calories: {calories} kcal\n"
        f"Protein: {protein}g\n"
        f"Fat: {fat}g\n"
        f"Carbs: {carbs}g\n"
        f"Tags: {', '.join(tags) if tags else 'N/A'}\n"
        f"Description: {description}"
    )


def evaluate_with_ragas(meal_plan, query, candidate_recipes, llm_output=None):
    """
    Simplified RAGAS evaluation for meal planning
    """
    
    # Prepare dataset
    data = prepare_ragas_dataset(meal_plan, query, candidate_recipes, llm_output)
    dataset = Dataset.from_dict(data)
    logger.info(f"Dataset created successfully with {len(dataset)} rows")
    
    # Configure LLM explicitly using LangChain ChatOpenAI
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        chat_llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            api_key=openai_api_key
        )
        llm = LangchainLLMWrapper(chat_llm)
        logger.info("Using ChatOpenAI (LangChain) wrapped for RAGAS metrics")
    else:
        llm = None
        logger.warning("OPENAI_API_KEY not set, RAGAS will use default LLM")
    
    # Instantiate ContextRelevance with LLM if available
    if llm:
        context_relevance_metric = ContextRelevance(llm=llm)
        logger.info("Created ContextRelevance with explicit LLM")
    else:
        context_relevance_metric = ContextRelevance()
        logger.warning("Created ContextRelevance without LLM (may fail)")
    
    metrics_to_use = [
        faithfulness,
        answer_relevancy,
        context_relevance_metric,
    ]
    
    # Set LLM on other metrics explicitly to bypass RAGAS internal LLM creation
    if llm:
        for metric in metrics_to_use:
            if isinstance(metric, ContextRelevance):
                continue
            if hasattr(metric, 'llm'):
                metric.llm = llm
            elif hasattr(metric, 'set_llm'):
                metric.set_llm(llm)
        logger.info("Explicitly configured LLM on all metrics to bypass internal creation")
    
    # Run evaluation
    logger.info("Running RAGAS evaluation...")
    try:
        results = evaluate(dataset, metrics=metrics_to_use)
        return results
    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {str(e)}", exc_info=True)
        return None



def evaluate_custom_metrics(meal_plan, query, candidate_recipes):
    """Quick custom metrics without LLM"""
    
    days = meal_plan.get("days", [])
    if not days:
        return {
            "macro_accuracy": 0.0,
            "meal_diversity": 0.0,
            "constraint_violations": 0
        }
    
    # Macro accuracy - get targets from query (no hardcoded defaults)
    errors = []
    target_calories = query.get("target_calories", 0)
    target_protein = query.get("target_protein", 0)
    target_fat = query.get("target_fat", 0)
    target_carbs = query.get("target_carbs", 0)
    
    for day_data in days:
        actual_calories = day_data.get("total_calories", 0)
        actual_protein = day_data.get("total_protein", 0)
        actual_fat = day_data.get("total_fat", 0)
        actual_carbs = day_data.get("total_carbs", 0)
        
        if target_calories > 0:
            error_pct = abs(actual_calories - target_calories) / target_calories * 100
            errors.append(error_pct)
        if target_protein > 0:
            error_pct = abs(actual_protein - target_protein) / target_protein * 100
            errors.append(error_pct)
        if target_fat > 0:
            error_pct = abs(actual_fat - target_fat) / target_fat * 100
            errors.append(error_pct)
        if target_carbs > 0:
            error_pct = abs(actual_carbs - target_carbs) / target_carbs * 100
            errors.append(error_pct)
    
    if errors:
        avg_error = sum(errors) / len(errors)
        macro_accuracy = max(0, 1 - avg_error / 100)
    else:
        macro_accuracy = 0.0
    
    # Meal diversity
    meal_ids = []
    for day_data in days:
        if hasattr(day_data, 'dict'):
            day_data = day_data.dict()
            
        meals = day_data.get("meals", [])
        for meal in meals:
            if meal is not None:
                if hasattr(meal, 'dict'):
                    meal = meal.dict()
                    
                recipe_id = meal.get("recipe_id") or meal.get("id")
                if recipe_id:
                    meal_ids.append(recipe_id)
    
    if meal_ids:
        unique_meals = len(set(meal_ids))
        meal_diversity = unique_meals / len(meal_ids)
    else:
        meal_diversity = 0.0
    
    # Constraint violations
    violations = 0
    exclusions = query.get("exclusions", [])
    
    if exclusions:
        if isinstance(exclusions, str):
            exclusions_list = [excl.strip().lower() for excl in exclusions.split(',') if excl.strip()]
        elif isinstance(exclusions, list):
            exclusions_list = [excl.strip().lower() if isinstance(excl, str) else str(excl).strip().lower() 
                             for excl in exclusions if excl]
        else:
            exclusions_list = []
        
        if exclusions_list:
            for day_data in days:
                if hasattr(day_data, 'dict'):
                    day_data = day_data.dict()
                    
                meals = day_data.get("meals", [])
                for meal in meals:
                    if meal is not None:
                        if hasattr(meal, 'dict'):
                            meal = meal.dict()
                            
                        ingredients = meal.get("ingredients") or meal.get("ingredients_raw", [])
                        if ingredients:
                            ingredient_str = " ".join([str(ing).lower() for ing in ingredients])
                            for exclusion in exclusions_list:
                                if exclusion and exclusion in ingredient_str:
                                    violations += 1
                                    break
    
    return {
        "macro_accuracy": macro_accuracy,
        "meal_diversity": meal_diversity,
        "constraint_violations": violations
    }


def get_grade(score):
    """Convert score to grade"""
    if score >= 0.9:
        return "A"
    elif score >= 0.8:
        return "B"
    elif score >= 0.7:
        return "C"
    elif score >= 0.6:
        return "D"
    else:
        return "F"



def comprehensive_ragas_evaluation(
    meal_plan,
    query,
    candidate_recipes,
    llm_output=None,
    include_custom_metrics=True
):
    """
    Complete evaluation using RAGAS + custom meal planning metrics
    """
    
    logger.info("=" * 80)
    logger.info("RAGAS EVALUATION FOR MEAL PLANNING")
    logger.info("=" * 80)
    
    # Standard RAGAS evaluation
    logger.info("\n[1/2] Running standard RAGAS metrics...")
    ragas_results = evaluate_with_ragas(meal_plan, query, candidate_recipes, llm_output)
    
    # Parse RAGAS results
    ragas_scores = {'faithfulness': 0, 'answer_relevancy': 0, 'context_relevancy': 0}
    if ragas_results and hasattr(ragas_results, 'to_pandas'):
        try:
            df = ragas_results.to_pandas()
            context_relevancy_col = 'nv_context_relevance' if 'nv_context_relevance' in df.columns else 'context_relevancy'
            ragas_scores = {
                'faithfulness': float(df['faithfulness'].iloc[0]) if 'faithfulness' in df.columns else 0,
                'answer_relevancy': float(df['answer_relevancy'].iloc[0]) if 'answer_relevancy' in df.columns else 0,
                'context_relevancy': float(df[context_relevancy_col].iloc[0]) if context_relevancy_col in df.columns else 0,
            }
        except Exception as e:
            logger.error(f"Error parsing RAGAS results: {str(e)}", exc_info=True)
    elif not ragas_results:
        logger.warning("RAGAS evaluation returned no results")
    
    # Handle NaN values
    faithfulness_val = ragas_scores.get('faithfulness', 0)
    answer_relevancy_val = ragas_scores.get('answer_relevancy', 0)
    context_relevancy_val = ragas_scores.get('context_relevancy', 0)
    
    if isinstance(faithfulness_val, float) and math.isnan(faithfulness_val):
        faithfulness_val = 0.0
    if isinstance(answer_relevancy_val, float) and math.isnan(answer_relevancy_val):
        answer_relevancy_val = 0.0
    if isinstance(context_relevancy_val, float) and math.isnan(context_relevancy_val):
        context_relevancy_val = 0.0
    
    logger.info("\n📊 RAGAS Results:")
    logger.info(f"Faithfulness: {faithfulness_val:.3f}")
    logger.info(f"Answer Relevancy: {answer_relevancy_val:.3f}")
    logger.info(f"Context Relevancy: {context_relevancy_val:.3f}")
    
    # Custom metrics
    if include_custom_metrics:
        logger.info("\n[2/2] Running custom meal planning metrics...")
        custom_results = evaluate_custom_metrics(meal_plan, query, candidate_recipes)
        
        logger.info("\n🍽️  Custom Metrics:")
        logger.info(f"Macro Accuracy: {custom_results['macro_accuracy']:.3f}")
        logger.info(f"Meal Diversity: {custom_results['meal_diversity']:.3f}")
        logger.info(f"Constraint Violations: {custom_results['constraint_violations']}")
    else:
        custom_results = {}
    
    ragas_score = (
        faithfulness_val * 0.3 +
        answer_relevancy_val * 0.3 +
        context_relevancy_val * 0.4
    )
    
    custom_score = (
        custom_results.get('macro_accuracy', 0) * 0.5 +
        custom_results.get('meal_diversity', 0) * 0.3 +
        (1 - min(custom_results.get('constraint_violations', 0) / 10, 1)) * 0.2
    )
    
    overall_score = ragas_score * 0.4 + custom_score * 0.6
    grade = get_grade(overall_score)
    
    logger.info("\n" + "-" * 80)
    logger.info("OVERALL SCORE".center(80))
    logger.info("-" * 80)
    logger.info(f"RAGAS Score: {ragas_score:.3f}")
    logger.info(f"Custom Score: {custom_score:.3f}")
    logger.info(f"Combined Score: {overall_score:.3f}")
    logger.info(f"Grade: {grade}")
    logger.info("=" * 80)
    
    return {
        "ragas": ragas_scores,
        "custom": custom_results,
        "overall_score": overall_score,
        "grade": grade
    }
