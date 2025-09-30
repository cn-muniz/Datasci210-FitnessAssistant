import sqlite3
import json
from datetime import datetime

def get_db_connection():
    conn = sqlite3.connect('fitplan.db')  # Adjust path if needed
    conn.row_factory = sqlite3.Row
    return conn

def read_grocery_lists():
    conn = get_db_connection()
    plans = []
    table_name = 'grocery_lists'
    plans = conn.execute(f'''
                SELECT gl.*, u.name as user_name 
                FROM {table_name} gl 
                LEFT JOIN users u ON gl.user_id = u.id 
                ORDER BY gl.created_at DESC
                ''').fetchall()
    
    conn.close()
    
    print("=" * 50)
    if table_name:
        print(f"GROCERY LISTS TABLE ({table_name}) - {len(plans)} records found")
    else:
        print("GROCERY LISTS TABLE - Table not found")
        print("Tried table names: grocery_lists, grocery_plans, shopping_lists")
        return
    print("=" * 50)
    
    if not plans:
        print("No grocery lists found in database.")
        return
    
    for plan in plans:
        print(f"\nList ID: {plan['id']}")
        print(f"User: {plan['user_name']} (ID: {plan['user_id']})")
        print(f"Created: {plan['created_at']}")
        
        # Handle different possible column names
        data_column = None
        for col in ['plan_data', 'list_data', 'grocery_data']:
            if col in plan.keys() and plan[col]:
                data_column = col
                break
        
        if data_column and plan[data_column]:
            try:
                plan_data = json.loads(plan[data_column])
                print(f"Grocery Data:")
                print(json.dumps(plan_data, indent=2))
            except json.JSONDecodeError:
                print(f"Grocery Data (raw): {plan[data_column]}")
        else:
            print("Grocery Data: None")
        
        print("-" * 50)

if __name__ == "__main__":
    read_grocery_lists()