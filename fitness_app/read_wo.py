import sqlite3
import json
from datetime import datetime

def get_db_connection():
    conn = sqlite3.connect('fitplan.db')  # Adjust path if needed
    conn.row_factory = sqlite3.Row
    return conn

def read_workout_plans():
    conn = get_db_connection()
    plans = conn.execute('''
        SELECT wp.*, u.name as user_name 
        FROM workout_plans wp 
        LEFT JOIN users u ON wp.user_id = u.id 
        ORDER BY wp.created_at DESC
    ''').fetchall()
    conn.close()
    
    print("=" * 50)
    print(f"WORKOUT PLANS TABLE - {len(plans)} records found")
    print("=" * 50)
    
    if not plans:
        print("No workout plans found in database.")
        return
    
    for plan in plans:
        print(f"\nPlan ID: {plan['id']}")
        print(f"User: {plan['user_name']} (ID: {plan['user_id']})")
        print(f"Created: {plan['created_at']}")
        
        if plan['plan_data']:
            try:
                plan_data = json.loads(plan['plan_data'])
                print(f"Plan Data:")
                print(json.dumps(plan_data, indent=2))
            except json.JSONDecodeError:
                print(f"Plan Data (raw): {plan['plan_data']}")
        else:
            print("Plan Data: None")
        
        print("-" * 50)

if __name__ == "__main__":
    read_workout_plans()