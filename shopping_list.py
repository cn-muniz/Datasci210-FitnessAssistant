import cohere
import os
import json

co = cohere.Client(os.environ.get('COHERE_API_KEY'))

def generate_shopping_list(recipes):
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
- Normalize plural/singular to a singular concept for the name (e.g., “onions” → “onion”) but keep quantities correct.

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
  • **Oils**: keep separate by base → olive oil, canola oil, vegetable oil, avocado oil; (extra-virgin vs light may be merged to “olive oil” unless explicitly required)
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

    Now, here are the recipes as JSON:
    {json.dumps(recipes, indent=2)}
    """

    response = co.chat(
        model="command-a-03-2025",
        message=user_message,
    )

    # Print and return model output
    print(response.text)
    return response.text


if __name__ == "__main__":
    # Test Case 1: Focuses on name normalization and aliases
    test_case_1 = [
        # Name variations and aliases
        ["2 cups all purpose flour", "1.5 cups AP flour", "2 cups plain flour"],
        ["3 scallions", "2 green onions", "4 large scallions"],
        ["2 tbsp EVOO", "3 tbsp extra virgin olive oil", "1/4 cup olive oil"],
        ["1 can garbanzo beans", "2 cans chickpeas", "400g chick peas"],
        ["500g minced beef", "1 lb ground beef", "250g beef (ground)"],
        ["2 cups caster sugar", "1 cup granulated sugar", "1.5 cups white sugar"],
        ["3 chicken breasts", "2 boneless skinless chicken breasts", "4 skinless chicken breasts"],
        # Mix of units and descriptors
        ["2-3 large onions", "1 medium white onion", "2 small red onions"],
        ["4 tbsp fresh chopped cilantro", "1/2 cup coriander leaves", "3 tbsp minced cilantro"],
        ["2 cloves garlic", "1 large garlic clove", "3 minced garlic cloves"],
        ["1 cup powdered sugar", "200g icing sugar", "1.5 cups confectioners sugar"],
        ["2 tsp bicarb soda", "1 tsp bicarbonate of soda", "1.5 tsp baking soda"],
        ["500 ml whole milk", "2 cups 2% milk", "1 pint skim milk"],
        ["3 large eggs", "4 medium eggs", "2 fresh eggs"],
        ["2 tbsp dried basil", "1/4 cup fresh basil", "3 tbsp chopped fresh basil"],
        ["1 cup packed brown sugar", "200g dark brown sugar", "150g light brown sugar"],
        ["2 cans diced tomatoes", "3 fresh tomatoes", "400g crushed tomatoes"],
        ["1 lb unsalted butter", "500g salted butter", "2 sticks butter"],
        ["3 tbsp low-sodium soy sauce", "2 tbsp regular soy sauce", "4 tbsp reduced-sodium soy sauce"],
        ["2 heads lettuce", "3 small heads iceberg lettuce", "1 large head romaine lettuce"]
    ]

    # Test Case 2: Focuses on unit conversions and ranges
    test_case_2 = [
        # Volume conversions
        ["1 cup water", "8 fl oz water", "240 ml water"],
        ["2 tbsp vanilla extract", "6 tsp vanilla extract", "1 fl oz vanilla extract"],
        ["1 quart heavy cream", "2 pints heavy cream", "4 cups heavy cream"],
        ["3-4 cups vegetable stock", "1 liter vegetable broth", "2 pints vegetable stock"],
        ["2 tbsp + 1 tsp honey", "35 ml honey", "1.5 fl oz honey"],
        # Mass conversions
        ["500g rice", "1 lb rice", "16 oz rice"],
        ["2.5 kg potatoes", "5 lbs potatoes", "80 oz potatoes"],
        ["100g chocolate chips", "4 oz chocolate chips", "0.25 lb chocolate chips"],
        ["750g pasta", "1.5 lbs pasta", "24 oz pasta"],
        ["50g butter", "2 oz butter", "0.125 lb butter"],
        # Count units and ranges
        ["2-3 bell peppers", "1 large bell pepper", "4 small bell peppers"],
        ["3-4 cloves garlic", "2 large garlic cloves", "5 small garlic cloves"],
        ["1 bunch parsley", "2 small bunches parsley", "1.5 bunches fresh parsley"],
        ["2 cans tuna", "3 small cans tuna", "1 large can tuna"],
        ["4-5 carrots", "3 medium carrots", "6 small carrots"],
        ["2 packages cream cheese", "16 oz cream cheese", "500g cream cheese"],
        ["3 sticks cinnamon", "2-3 cinnamon sticks", "4 large cinnamon sticks"],
        ["1 head cauliflower", "2 small heads cauliflower", "1.5 medium heads cauliflower"],
        ["2-3 lemons", "4 small lemons", "1 large lemon"],
        ["5-6 mushrooms", "8 medium mushrooms", "4 large mushrooms"]
    ]

    # Test Case 3: Focuses on edge cases and ambiguous measurements
    test_case_3 = [
        # Ambiguous measurements
        ["1 jar marinara sauce", "2 small jars pasta sauce", "1 large jar tomato sauce"],
        ["1 container yogurt", "500g yogurt", "2 cups yogurt"],
        ["1 package spinach", "10 oz frozen spinach", "300g fresh spinach"],
        ["1 bottle olive oil", "500ml olive oil", "2 cups olive oil"],
        ["1 bag rice", "2 lb rice", "1 kg rice"],
        # Mixed units for same ingredient
        ["2 cups shredded cheese", "250g grated cheese", "8 oz cheese"],
        ["1 cup diced onions", "200g chopped onions", "2 medium onions"],
        ["500ml coconut milk", "2 cups coconut milk", "1 can coconut milk"],
        ["3 tbsp dried oregano", "1/4 cup fresh oregano", "15g oregano"],
        ["1 lb ground pork", "500g minced pork", "2 packages ground pork"],
        # Descriptors and preparations
        ["2 cups peeled diced potatoes", "4 medium russet potatoes", "500g baby potatoes"],
        ["1 cup sliced mushrooms", "8 oz whole mushrooms", "4 large portobello mushrooms"],
        ["2 tbsp minced ginger", "30g fresh ginger root", "3-inch piece ginger"],
        ["3 cups shredded cabbage", "1 medium head cabbage", "500g chopped cabbage"],
        ["2 tbsp crushed garlic", "6 cloves minced garlic", "30g garlic paste"],
        # Regional variations
        ["2 cups aubergine", "1 large eggplant", "500g brinjal"],
        ["3 spring onions", "4 green onions", "2 bunches scallions"],
        ["1 tin tomatoes", "400g canned tomatoes", "14 oz diced tomatoes"],
        ["2 courgettes", "3 medium zucchini", "400g summer squash"],
        ["100g rocket", "2 cups arugula", "1 bunch arugula"]
    ]

    # Usage example:
    generate_shopping_list(test_case_2)