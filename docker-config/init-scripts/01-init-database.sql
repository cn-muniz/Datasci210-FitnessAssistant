-- FitPlan Database Initialization Script
-- This script runs automatically when PostgreSQL container starts for the first time

-- Enable UUID extension (useful for generating unique IDs)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create schemas if needed
CREATE SCHEMA IF NOT EXISTS public;

-- Grant privileges
GRANT ALL PRIVILEGES ON SCHEMA public TO fitplan_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO fitplan_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO fitplan_user;

-- Create users table (migrated from SQLite)
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
    bmr DECIMAL(10,2),
    tdee DECIMAL(10,2),
    caloric_target DECIMAL(10,2),
    protein_target_g DECIMAL(10,2),
    carbs_target_g DECIMAL(10,2),
    fat_target_g DECIMAL(10,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create workout_plans table (migrated from SQLite)
CREATE TABLE IF NOT EXISTS workout_plans (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    week_date DATE NOT NULL,
    plan_data JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Create meal_plans table (migrated from SQLite)
CREATE TABLE IF NOT EXISTS meal_plans (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    week_date DATE NOT NULL,
    plan_data JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Create grocery_lists table (migrated from SQLite)
CREATE TABLE IF NOT EXISTS grocery_lists (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    week_date DATE NOT NULL,
    grocery_data JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_workout_plans_user_id ON workout_plans(user_id);
CREATE INDEX IF NOT EXISTS idx_workout_plans_week_date ON workout_plans(week_date);
CREATE INDEX IF NOT EXISTS idx_meal_plans_user_id ON meal_plans(user_id);
CREATE INDEX IF NOT EXISTS idx_meal_plans_week_date ON meal_plans(week_date);
CREATE INDEX IF NOT EXISTS idx_grocery_lists_user_id ON grocery_lists(user_id);
CREATE INDEX IF NOT EXISTS idx_grocery_lists_week_date ON grocery_lists(week_date);

-- Insert a test user (optional - for development)
INSERT INTO users (name, email, password, age, gender, weight, height, activity_level, fitness_goals)
VALUES 
    ('Test User', 'test@fitplan.com', 'hashed_password_here', 30, 'male', 180.0, 70.0, 'moderate', 'Weight Loss')
ON CONFLICT (email) DO NOTHING;

-- Log completion
DO $$
BEGIN
    RAISE NOTICE 'FitPlan database initialized successfully!';
END $$;
