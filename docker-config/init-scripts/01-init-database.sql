-- ============================================================================
-- FitPlan Database Initialization Script - v2.0
-- Dynamic Start Date Support with Timezone Awareness
-- This script runs automatically when PostgreSQL container starts
-- ============================================================================

-- Enable UUID extension (useful for generating unique IDs)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create schemas if needed
CREATE SCHEMA IF NOT EXISTS public;

-- Grant privileges
GRANT ALL PRIVILEGES ON SCHEMA public TO fitplan_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO fitplan_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO fitplan_user;

-- ============================================================================
-- Create users table with timezone support
-- ============================================================================
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    age INTEGER,
    gender VARCHAR(50),
    weight DECIMAL(5,2),
    height DECIMAL(5,2),
    activity_level VARCHAR(50),
    fitness_goals TEXT,
    dietary_restrictions TEXT,
    physical_limitations TEXT,
    available_equipment TEXT,
    workout_schedule VARCHAR(10),  -- Added: workout frequency preference
    food_preferences TEXT,  -- Added: food preferences
    food_exclusions TEXT,  -- Added: foods to avoid
    bmr DECIMAL(10,2),
    tdee DECIMAL(10,2),
    caloric_target DECIMAL(10,2),
    protein_target_g DECIMAL(10,2),
    carbs_target_g DECIMAL(10,2),
    fat_target_g DECIMAL(10,2),
    timezone VARCHAR(50) DEFAULT 'UTC',  -- NEW: User timezone
    privacy_accepted BOOLEAN DEFAULT FALSE,  -- NEW: Privacy policy acceptance
    privacy_accepted_at TIMESTAMP,  -- NEW: When privacy was accepted
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- Create workout_plans table with start_date (renamed from week_date)
-- ============================================================================
CREATE TABLE IF NOT EXISTS workout_plans (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    start_date DATE NOT NULL,  -- CHANGED: renamed from week_date
    plan_data JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP,  -- NEW: track plan updates
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    UNIQUE (user_id, start_date)  -- CHANGED: ensure one plan per user per start_date
);

-- ============================================================================
-- Create meal_plans table with start_date (renamed from week_date)
-- ============================================================================
CREATE TABLE IF NOT EXISTS meal_plans (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    start_date DATE NOT NULL,  -- CHANGED: renamed from week_date
    plan_data JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP,  -- NEW: track plan updates
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    UNIQUE (user_id, start_date)  -- CHANGED: ensure one plan per user per start_date
);

-- ============================================================================
-- Create grocery_lists table with start_date (renamed from week_date)
-- ============================================================================
CREATE TABLE IF NOT EXISTS grocery_lists (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    start_date DATE NOT NULL,  -- CHANGED: renamed from week_date
    grocery_data JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP,  -- NEW: track list updates
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    UNIQUE (user_id, start_date)  -- CHANGED: ensure one list per user per start_date
);

-- ============================================================================
-- Create indexes for better query performance
-- ============================================================================

-- Users indexes
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_timezone ON users(timezone);  -- NEW

-- Workout plans indexes
CREATE INDEX IF NOT EXISTS idx_workout_plans_user_id ON workout_plans(user_id);
CREATE INDEX IF NOT EXISTS idx_workout_plans_start_date ON workout_plans(start_date);  -- CHANGED
CREATE INDEX IF NOT EXISTS idx_workout_plans_user_start ON workout_plans(user_id, start_date);  -- NEW
CREATE INDEX IF NOT EXISTS idx_workout_plans_created_at ON workout_plans(created_at);  -- NEW

-- Meal plans indexes
CREATE INDEX IF NOT EXISTS idx_meal_plans_user_id ON meal_plans(user_id);
CREATE INDEX IF NOT EXISTS idx_meal_plans_start_date ON meal_plans(start_date);  -- CHANGED
CREATE INDEX IF NOT EXISTS idx_meal_plans_user_start ON meal_plans(user_id, start_date);  -- NEW
CREATE INDEX IF NOT EXISTS idx_meal_plans_created_at ON meal_plans(created_at);  -- NEW

-- Grocery lists indexes
CREATE INDEX IF NOT EXISTS idx_grocery_lists_user_id ON grocery_lists(user_id);
CREATE INDEX IF NOT EXISTS idx_grocery_lists_start_date ON grocery_lists(start_date);  -- CHANGED
CREATE INDEX IF NOT EXISTS idx_grocery_lists_user_start ON grocery_lists(user_id, start_date);  -- NEW
CREATE INDEX IF NOT EXISTS idx_grocery_lists_created_at ON grocery_lists(created_at);  -- NEW


-- ============================================================================
-- Insert test users (optional - for development)
-- ============================================================================

INSERT INTO users (
    name, 
    email, 
    password, 
    age, 
    gender, 
    weight, 
    height, 
    activity_level, 
    fitness_goals,
    timezone,
    privacy_accepted,
    privacy_accepted_at
)
VALUES 
    (
        'Test User', 
        'test@fitplan.com', 
        'hashed_password_here', 
        30, 
        'male', 
        180.0, 
        70.0, 
        'moderately_active', 
        'weight-loss',
        'America/Los_Angeles',
        TRUE,
        CURRENT_TIMESTAMP
    ),
    (
        'Demo User', 
        'demo@fitplan.com', 
        'hashed_password_here', 
        25, 
        'female', 
        140.0, 
        65.0, 
        'lightly_active', 
        'muscle-building',
        'America/New_York',
        TRUE,
        CURRENT_TIMESTAMP
    )
ON CONFLICT (email) DO NOTHING;

-- ============================================================================
-- Grant permissions on new objects
-- ============================================================================

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO fitplan_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO fitplan_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO fitplan_user;

-- Final success message
DO $$
BEGIN
    RAISE NOTICE '==================================================';
    RAISE NOTICE 'Database ready for FitPlan application!';
    RAISE NOTICE 'Time zone support: Enabled';
    RAISE NOTICE 'Dynamic start dates: Enabled';
    RAISE NOTICE '==================================================';
END $$;