from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
import sqlite3
import hashlib
import json
import os
from datetime import datetime, timedelta

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production

# Database setup
def init_db():
    conn = sqlite3.connect('fitplan.db')
    c = conn.cursor()
    
    # Users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            weight REAL,
            height TEXT,
            fitness_goals TEXT,
            dietary_restrictions TEXT,
            physical_limitations TEXT,
            available_equipment TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Meal plans table
    c.execute('''
        CREATE TABLE IF NOT EXISTS meal_plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            week_date TEXT,
            plan_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Workout plans table
    c.execute('''
        CREATE TABLE IF NOT EXISTS workout_plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            week_date TEXT,
            plan_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    # Grocery data table
    c.execute('''
        CREATE TABLE IF NOT EXISTS grocery_lists (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            week_date TEXT,
            grocery_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    
    conn.commit()
    conn.close()

# Register the fromjson filter
@app.template_filter('fromjson')
def fromjson_filter(value):
    if value is None:
        return []
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return []
    return value

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def get_db_connection():
    conn = sqlite3.connect('fitplan.db')
    conn.row_factory = sqlite3.Row
    return conn

# Routes
@app.route('/')
def index():
    return render_template('welcome.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/register', methods=['POST'])
def register():
    name = request.form['name']
    email = request.form['email']
    password = request.form['password']
    confirm_password = request.form['confirm_password']
    
    if password != confirm_password:
        flash('Passwords do not match')
        return redirect(url_for('signup'))
    
    conn = get_db_connection()
    try:
        conn.execute('INSERT INTO users (name, email, password) VALUES (?, ?, ?)',
                    (name, email, hash_password(password)))
        conn.commit()
        
        # Get the user ID
        user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        session['user_id'] = user['id']
        session['user_name'] = user['name']
        
        conn.close()
        return redirect(url_for('questionnaire_intro'))
    except sqlite3.IntegrityError:
        flash('Email already exists')
        conn.close()
        return redirect(url_for('signup'))

@app.route('/login', methods=['POST'])
def login():
    email = request.form['email']
    password = request.form['password']
    
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE email = ? AND password = ?',
                       (email, hash_password(password))).fetchone()
    conn.close()
    
    if user:
        session['user_id'] = user['id']
        session['user_name'] = user['name']
        return redirect(url_for('dashboard'))
    else:
        flash('Invalid email or password')
        return redirect(url_for('index'))

@app.route('/questionnaire-intro')
def questionnaire_intro():
    if 'user_id' not in session:
        return redirect(url_for('index'))
    return render_template('questionnaire_intro.html')

@app.route('/basic-info')
def basic_info():
    if 'user_id' not in session:
        return redirect(url_for('index'))
    return render_template('basic_info.html')

@app.route('/save-basic-info', methods=['POST'])
def save_basic_info():
    if 'user_id' not in session:
        return redirect(url_for('index'))
    
    gender = request.form['gender']
    age = request.form['age']
    height = request.form['height']
    weight = request.form['weight']
    
    conn = get_db_connection()
    conn.execute('UPDATE users SET gender = ?, age = ?, height = ?, weight = ? WHERE id = ?',
                (gender, age, height, weight, session['user_id']))
    conn.commit()
    conn.close()
    
    return redirect(url_for('fitness_goals'))

@app.route('/fitness-goals')
def fitness_goals():
    if 'user_id' not in session:
        return redirect(url_for('index'))
    return render_template('fitness_goals.html')

@app.route('/save-fitness-goals', methods=['POST'])
def save_fitness_goals():
    if 'user_id' not in session:
        return redirect(url_for('index'))
    
    goals = request.form['fitness_goal']
    
    conn = get_db_connection()
    conn.execute('UPDATE users SET fitness_goals = ? WHERE id = ?',
                (goals, session['user_id']))
    conn.commit()
    conn.close()
    
    return redirect(url_for('dietary_restrictions'))

@app.route('/dietary-restrictions')
def dietary_restrictions():
    if 'user_id' not in session:
        return redirect(url_for('index'))
    return render_template('dietary_restrictions.html')

@app.route('/save-dietary-restrictions', methods=['POST'])
def save_dietary_restrictions():
    if 'user_id' not in session:
        return redirect(url_for('index'))
    
    restrictions = request.form.getlist('dietary_restrictions')
    restrictions_json = json.dumps(restrictions)
    
    conn = get_db_connection()
    conn.execute('UPDATE users SET dietary_restrictions = ? WHERE id = ?',
                (restrictions_json, session['user_id']))
    conn.commit()
    conn.close()
    
    return redirect(url_for('physical_limitations'))

@app.route('/physical-limitations')
def physical_limitations():
    if 'user_id' not in session:
        return redirect(url_for('index'))
    return render_template('physical_limitations.html')

@app.route('/save-physical-limitations', methods=['POST'])
def save_physical_limitations():
    if 'user_id' not in session:
        return redirect(url_for('index'))
    
    limitations = request.form.getlist('physical_limitations')
    limitations_json = json.dumps(limitations)
    
    conn = get_db_connection()
    conn.execute('UPDATE users SET physical_limitations = ? WHERE id = ?',
                (limitations_json, session['user_id']))
    conn.commit()
    conn.close()
    
    return redirect(url_for('equipment_access'))

@app.route('/equipment-access')
def equipment_access():
    if 'user_id' not in session:
        return redirect(url_for('index'))
    return render_template('equipment_access.html')

@app.route('/save-equipment-access', methods=['POST'])
def save_equipment_access():
    if 'user_id' not in session:
        return redirect(url_for('index'))
    
    equipment = request.form.getlist('equipment')
    equipment_json = json.dumps(equipment)
    
    conn = get_db_connection()
    conn.execute('UPDATE users SET available_equipment = ? WHERE id = ?',
                (equipment_json, session['user_id']))
    conn.commit()
    conn.close()
    
    return redirect(url_for('profile_summary'))

@app.route('/profile-summary')
def profile_summary():
    if 'user_id' not in session:
        return redirect(url_for('index'))
    
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE id = ?', 
                       (session['user_id'],)).fetchone()
    conn.close()
    
    return render_template('profile_summary.html', user=user)

@app.route('/create-plan', methods=['POST'])
def create_plan():
    if 'user_id' not in session:
        return redirect(url_for('index'))
    
    # This will call your ML pipeline later
    # For now, just create placeholder plans
    user_id = session['user_id']
    week_date = datetime.now().strftime('%Y-%m-%d')
    
    # Sample workout plan data
    workout_data = {
        "week": "March 18-24",
        "days": [
            {
                "day": "Monday",
                "title": "Upper Body Push",
                "duration": "45 min",
                "exercises": [
                    {"name": "Dumbbell Chest Press", "sets": "3 × 12"},
                    {"name": "Dumbbell Shoulder Press", "sets": "3 × 10"},
                    {"name": "Dumbbell Tricep Extensions", "sets": "3 × 12"},
                    {"name": "Lateral Raises", "sets": "3 × 15"}
                ]
            },
            {
                "day": "Tuesday",
                "title": "Rest Day",
                "duration": "—",
                "description": "Recovery day. Light walking or gentle stretching recommended."
            },
            {
                "day": "Wednesday", 
                "title": "Upper Body Pull",
                "duration": "40 min",
                "exercises": [
                    {"name": "Dumbbell Bent-Over Rows", "sets": "3 × 12"},
                    {"name": "Single-Arm Rows", "sets": "3 × 10 each"},
                    {"name": "Dumbbell Bicep Curls", "sets": "3 × 15"},
                    {"name": "Reverse Flyes", "sets": "3 × 12"}
                ]
            },
            {
                "day": "Thursday",
                "title": "Rest Day", 
                "duration": "—",
                "description": "Active recovery. 20-30 minutes of walking recommended."
            },
            {
                "day": "Friday",
                "title": "Lower Body",
                "duration": "35 min",
                "exercises": [
                    {"name": "Bodyweight Squats", "sets": "3 × 15"},
                    {"name": "Dumbbell Lunges", "sets": "3 × 10 each"},
                    {"name": "Seated Calf Raises", "sets": "3 × 20"},
                    {"name": "Glute Bridges", "sets": "3 × 15"}
                ]
            },
            {
                "day": "Saturday",
                "title": "Core & Cardio",
                "duration": "30 min", 
                "exercises": [
                    {"name": "Modified Planks", "sets": "3 × 30s"},
                    {"name": "Seated Russian Twists", "sets": "3 × 20"},
                    {"name": "Dead Bugs", "sets": "3 × 10 each"}
                ]
            },
            {
                "day": "Sunday",
                "title": "Rest Day",
                "duration": "—",
                "description": "Complete rest or light yoga/stretching."
            }
        ]
    }
    
    # Sample meal plan data
    meal_data = {
        "week": "March 18-24",
        "daily_calories": 1650,
        "days": {
            "Monday": {
                "date": "March 18",
                "meals": [
                    {
                        "title": "🌅 Breakfast",
                        "calories": 420,
                        "description": "Greek Yogurt Parfait with gluten-free granola and blueberries",
                        "macros": {"protein": "25g", "carbs": "45g", "fat": "18g"}
                    },
                    {
                        "title": "🥗 Lunch", 
                        "calories": 480,
                        "description": "Grilled Chicken Quinoa Bowl with roasted vegetables",
                        "macros": {"protein": "32g", "carbs": "42g", "fat": "16g"}
                    },
                    {
                        "title": "🍽️ Dinner",
                        "calories": 520,
                        "description": "Baked Salmon with sweet potato and steamed broccoli", 
                        "macros": {"protein": "35g", "carbs": "38g", "fat": "22g"}
                    },
                    {
                        "title": "🥜 Snacks",
                        "calories": 230,
                        "description": "Apple with almond butter, herbal tea",
                        "macros": {"protein": "8g", "carbs": "22g", "fat": "14g"}
                    }
                ]
            }
        }
    }
    
    # Sample grocery list
    grocery_data = {
        "week": "March 18-24",
        "sections": [
            {
                "title": "🥬 Produce",
                "items": [
                    {"name": "Blueberries", "quantity": "2 cups", "price": "$4.99"},
                    {"name": "Broccoli crowns", "quantity": "2 heads", "price": "$3.50"},
                    {"name": "Sweet potatoes", "quantity": "3 medium", "price": "$2.99"},
                    {"name": "Apples", "quantity": "4 large", "price": "$3.99"}
                ]
            },
            {
                "title": "🥩 Protein", 
                "items": [
                    {"name": "Chicken breast", "quantity": "2 lbs", "price": "$8.99"},
                    {"name": "Salmon fillets", "quantity": "4 pieces", "price": "$16.99"},
                    {"name": "Greek yogurt (plain)", "quantity": "32 oz", "price": "$5.99"}
                ]
            },
            {
                "title": "🌾 Pantry",
                "items": [
                    {"name": "Quinoa", "quantity": "1 lb bag", "price": "$6.99"},
                    {"name": "GF granola", "quantity": "1 box", "price": "$7.49"},
                    {"name": "Almond butter", "quantity": "1 jar", "price": "$8.99"},
                    {"name": "Tahini", "quantity": "1 container", "price": "$6.99"}
                ]
            }
        ],
        "total": "$89.50"
    }
    
    conn = get_db_connection()
    
    # Save workout plan
    conn.execute('INSERT INTO workout_plans (user_id, week_date, plan_data) VALUES (?, ?, ?)',
                (user_id, week_date, json.dumps(workout_data)))
    
    # Save meal plan  
    conn.execute('INSERT INTO meal_plans (user_id, week_date, plan_data) VALUES (?, ?, ?)',
                (user_id, week_date, json.dumps(meal_data)))
    
    # Store grocery data
    conn.execute('INSERT INTO grocery_lists (user_id, week_date, grocery_data) VALUES (?, ?, ?)',
                (user_id, week_date, json.dumps(grocery_data)))
    
    conn.commit()
    conn.close()
    
    
    
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('index'))
    
    conn = get_db_connection()
    
    # Get latest plans
    workout_plan = conn.execute('SELECT * FROM workout_plans WHERE user_id = ? ORDER BY created_at DESC LIMIT 1',
                               (session['user_id'],)).fetchone()
    
    meal_plan = conn.execute('SELECT * FROM meal_plans WHERE user_id = ? ORDER BY created_at DESC LIMIT 1', 
                            (session['user_id'],)).fetchone()
    
    grocery_list = conn.execute('SELECT * FROM grocery_lists WHERE user_id = ? ORDER BY created_at DESC LIMIT 1', 
                            (session['user_id'],)).fetchone()
    conn.close()
    
    workout_data = json.loads(workout_plan['plan_data']) if workout_plan else None
    meal_data = json.loads(meal_plan['plan_data']) if meal_plan else None
    grocery_data = json.loads(grocery_list['grocery_data']) if grocery_list else None
    
    # Extract first day from meal_data for dashboard preview
    first_day = None
    if meal_data:
        # Handle different possible structures of meal_data
        if isinstance(meal_data, list) and len(meal_data) > 0:
            # If meal_data is a list of days
            first_day = meal_data[0]
        elif isinstance(meal_data, dict):
            # If meal_data is a dict, check for common keys
            if 'days' in meal_data and meal_data['days']:
                first_day_key = next(iter(meal_data['days']))  # Gets first key
                first_day = meal_data['days'][first_day_key]
                # Add the day name to the first_day object for display
                first_day['day_name'] = first_day_key
            elif 'meal_plan' in meal_data and meal_data['meal_plan']:
                if isinstance(meal_data['meal_plan'], list):
                    first_day = meal_data['meal_plan'][0]
                else:
                    # If meal_plan is also a dict
                    first_day_key = next(iter(meal_data['meal_plan']))
                    first_day = meal_data['meal_plan'][first_day_key]
            elif 'day_1' in meal_data:
                first_day = meal_data['day_1']
            else:
                # If it's a single day object, use it directly
                first_day = meal_data
    
    return render_template('dashboard.html', 
                         workout_plan=workout_data,
                         meal_plan=meal_data,
                         first_day=first_day,  # Add this new variable
                         grocery_list=grocery_data,
                         user_name=session.get('user_name'))

# Placeholder API endpoints for plan generation
@app.route('/api/generate-workout-plan', methods=['POST'])
def generate_workout_plan():
    return jsonify({"message": "Workout plan generation - Work in progress"})

@app.route('/api/generate-meal-plan', methods=['POST']) 
def generate_meal_plan():
    return jsonify({"message": "Meal plan generation - Work in progress"})

@app.route('/api/generate-grocery-list', methods=['POST'])
def generate_grocery_list():
    return jsonify({"message": "Grocery list generation - Work in progress"})

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

# if __name__ == '__main__':
#     init_db()
#     app.run(debug=True)
if __name__ == '__main__':
    init_db()
    port = int(os.environ.get('PORT', 5000))
    # Enable debug mode only in local development
    debug_mode = os.environ.get('RENDER') is None  # Render sets this env var
    app.run(host='0.0.0.0', port=port, debug=debug_mode)