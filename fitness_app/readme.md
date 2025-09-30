# FitPlan Flask Prototype

A full-stack web application prototype for personalized health and fitness planning, built with Flask and SQLite.

## Features

### Completed ✅
- **User Authentication**: Registration and login system
- **Onboarding Flow**: 6-step questionnaire matching your wireframes
- **Database Integration**: SQLite with user profiles and plans
- **Responsive Design**: Mobile-first UI matching wireframe aesthetics
- **Interactive Dashboard**: Tabbed interface for workouts, meals, and grocery lists
- **Progress Tracking**: Local storage for workout completion and meal logging
- **Form Validation**: Client and server-side validation
- **Sample Data**: Pre-populated workout plans, meal plans, and grocery lists

### Work in Progress 🚧
- ML Pipeline Integration (placeholder endpoints created)
- Advanced meal plan generation
- Workout plan customization
- Restaurant recommendations

## File Structure

```
fitplan-app/
├── app.py                          # Main Flask application
├── fitplan.db                      # SQLite database (auto-created)
├── requirements.txt                # Python dependencies
├── static/
│   ├── css/
│   │   ├── main.css               # Global styles
│   │   ├── forms.css              # Onboarding forms
│   │   └── dashboard.css          # Dashboard & plans
│   └── js/
│       ├── main.js                # Core functionality
│       ├── forms.js               # Form handling
│       └── plans.js               # Dashboard interactions
└── templates/
    ├── base.html                  # Common layout
    ├── welcome.html               # Login page
    ├── signup.html                # Registration
    ├── questionnaire_intro.html   # Onboarding start
    ├── basic_info.html            # Step 1: Basic info
    ├── fitness_goals.html         # Step 2: Goals
    ├── dietary_restrictions.html  # Step 3: Diet restrictions
    ├── physical_limitations.html  # Step 4: Limitations
    ├── equipment_access.html      # Step 5: Equipment
    ├── profile_summary.html       # Step 6: Summary
    └── dashboard.html             # Main dashboard
```

## Setup Instructions

### Using Docker

Need docker and docker-compose. Run

```bash
docker-compose up -d --build
```

Should be able to access at http://localhost:5001

### Prerequisites
- Python 3.7+
- pip (Python package manager)

### Installation

1. **Create project directory:**
   ```bash
   mkdir fitplan-app
   cd fitplan-app
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   
   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install flask sqlite3
   ```

4. **Create all the files** from the artifacts above in their respective directories.

5. **Run the application:**
   ```bash
   python app.py
   ```

6. **Open your browser** and navigate to:
   ```
   http://localhost:5000
   ```

## Usage Guide

### 1. Registration Flow
- Start at the welcome page
- Click "Create Account"
- Fill in basic registration info
- Complete the 6-step onboarding questionnaire

### 2. Onboarding Steps
1. **Basic Info**: Age, gender, height, weight
2. **Fitness Goals**: Weight loss, strength, muscle building, endurance
3. **Dietary Restrictions**: Vegetarian, gluten-free, allergies, etc.
4. **Physical Limitations**: Injuries or health considerations
5. **Equipment Access**: Available workout equipment
6. **Profile Summary**: Review and confirm details

### 3. Dashboard Features
- **Workout Tab**: Weekly workout schedule with exercise details
- **Meals Tab**: Daily meal plans with calorie and macro information
- **Grocery Tab**: Auto-generated shopping list organized by store section
- **Interactive Elements**: 
  - Click grocery items to check them off
  - Mark workouts as complete
  - Log meals as consumed

## Technical Details

### Database Schema
```sql
-- User profiles with all onboarding data
users: id, name, email, password, age, gender, weight, height, 
       fitness_goals, dietary_restrictions, physical_limitations, 
       available_equipment, created_at

-- Generated workout plans (JSON data)
workout_plans: id, user_id, week_date, plan_data, created_at

-- Generated meal plans (JSON data)  
meal_plans: id, user_id, week_date, plan_data, created_at
```

### API Endpoints (Placeholders)
```
POST /api/generate-workout-plan   # Returns "Work in progress"
POST /api/generate-meal-plan      # Returns "Work in progress"  
POST /api/generate-grocery-list   # Returns "Work in progress"
```

### Sample Data
The prototype includes realistic sample data:
- **Workout Plan**: 7-day schedule with back-friendly exercises using dumbbells
- **Meal Plan**: Gluten-free meals totaling 1,650 calories/day
- **Grocery List**: Organized by store sections with prices (~$89 total)

## Next Steps for Production

1. **ML Integration**: Connect the placeholder API endpoints to your RAG pipeline and LLM services
2. **User Management**: Add password reset, email verification, profile editing
3. **Data Persistence**: Move from localStorage to database for user progress
4. **Advanced Features**: 
   - Multiple week plans
   - Exercise instruction videos
   - Recipe details with cooking instructions
   - Grocery store integration
5. **Performance**: Add caching, database optimization, CDN for assets
6. **Security**: Environment variables, HTTPS, input sanitization, rate limiting

## Testing

### Sample User Data
You can test the app with these values:
- **Name**: John Smith
- **Age**: 32, Male
- **Height**: 5'10"
- **Weight**: 180 lbs
- **Goal**: Weight Loss
- **Restrictions**: Gluten-Free
- **Limitations**: Back Problems
- **Equipment**: Bodyweight, Dumbbells

The app will generate the sample workout and meal plans automatically after onboarding.

## Troubleshooting

### Common Issues
- **Database errors**: Delete `fitplan.db` and restart the app to recreate
- **CSS not loading**: Check file paths and ensure static files are in correct directories
- **Form validation**: JavaScript must be enabled for client-side validation
- **Mobile display**: Test on actual devices or use browser dev tools mobile view

### Development Mode
The app runs in debug mode by default. For production:
1. Set `app.debug = False`
2. Use environment variables for sensitive data
3. Use a production WSGI server like Gunicorn