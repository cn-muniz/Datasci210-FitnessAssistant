import sqlite3
import json
from datetime import datetime

def get_db_connection():
    conn = sqlite3.connect('fitplan.db')  # Adjust path if needed
    conn.row_factory = sqlite3.Row
    return conn

def read_users():
    conn = get_db_connection()
    users = conn.execute('SELECT * FROM users ORDER BY created_at DESC').fetchall()
    conn.close()
    
    print("=" * 50)
    print(f"USERS TABLE - {len(users)} records found")
    print("=" * 50)
    
    if not users:
        print("No users found in database.")
        return
    
    for user in users:
        print(f"\nUser ID: {user['id']}")
        print(f"Name: {user['name']}")
        print(f"Email: {user['email']}")
        print(f"Created: {user['created_at']}")

        # Handle JSON fields if they exist
        if 'fitness_goals' in user.keys() and user['fitness_goals']:
            try:
                goals = json.loads(user['fitness_goals'])
                print(f"Fitness Goals: {goals}")
            except json.JSONDecodeError:
                print(f"Fitness Goals: {user['fitness_goals']} (raw)")

        if 'dietary_restrictions' in user.keys() and user['dietary_restrictions']:
            try:
                restrictions = json.loads(user['dietary_restrictions'])
                print(f"Dietary Restrictions: {restrictions}")
            except json.JSONDecodeError:
                print(f"Dietary Restrictions: {user['dietary_restrictions']} (raw)")

        if 'available_equipment' in user.keys() and user['available_equipment']:
            try:
                equipment = json.loads(user['available_equipment'])
                print(f"Equipment: {equipment}")
            except json.JSONDecodeError:
                print(f"Equipment: {user['available_equipment']} (raw)")

        if 'health_info' in user.keys() and user['health_info']:
            try:
                health = json.loads(user['health_info'])
                print(f"Health Info: {health}")
            except json.JSONDecodeError:
                print(f"Health Info: {user['health_info']} (raw)")
        
        print("-" * 30)

if __name__ == "__main__":
    read_users()