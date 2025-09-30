import sqlite3
import sys

def get_db_connection():
    conn = sqlite3.connect('fitplan.db')  # Adjust path if needed
    conn.row_factory = sqlite3.Row
    return conn

def check_user_exists(user_id):
    """Check if user exists and return user info"""
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    return user

def count_user_records(user_id):
    """Count records for this user in all tables"""
    conn = get_db_connection()
    
    counts = {}
    
    # Count meal plans
    counts['meal_plans'] = conn.execute(
        'SELECT COUNT(*) as count FROM meal_plans WHERE user_id = ?', 
        (user_id,)
    ).fetchone()['count']
    
    # Count workout plans
    counts['workout_plans'] = conn.execute(
        'SELECT COUNT(*) as count FROM workout_plans WHERE user_id = ?', 
        (user_id,)
    ).fetchone()['count']
    
    # Count grocery lists
    counts['grocery_lists'] = conn.execute(
        'SELECT COUNT(*) as count FROM grocery_lists WHERE user_id = ?', 
        (user_id,)
    ).fetchone()['count']
    
    conn.close()
    return counts

def delete_user_data(user_id):
    """Delete all user data from all tables"""
    conn = get_db_connection()
    
    try:
        # Start transaction
        conn.execute('BEGIN TRANSACTION')
        
        # Delete meal plans
        meal_result = conn.execute(
            'DELETE FROM meal_plans WHERE user_id = ?', 
            (user_id,)
        )
        print(f"✓ Deleted {meal_result.rowcount} meal plan(s)")
        
        # Delete workout plans
        workout_result = conn.execute(
            'DELETE FROM workout_plans WHERE user_id = ?', 
            (user_id,)
        )
        print(f"✓ Deleted {workout_result.rowcount} workout plan(s)")
        
        # Delete grocery lists
        grocery_result = conn.execute(
            'DELETE FROM grocery_lists WHERE user_id = ?', 
            (user_id,)
        )
        print(f"✓ Deleted {grocery_result.rowcount} grocery list(s)")
        
        # Delete user
        user_result = conn.execute(
            'DELETE FROM users WHERE id = ?', 
            (user_id,)
        )
        print(f"✓ Deleted user record")
        
        # Commit transaction
        conn.commit()
        print("\n✅ All deletions completed successfully!")
        
    except Exception as e:
        # Rollback on error
        conn.rollback()
        print(f"\n❌ Error occurred: {e}")
        print("All changes have been rolled back.")
        return False
    
    finally:
        conn.close()
    
    return True

def main():
    # Get user ID from command line argument or prompt
    if len(sys.argv) > 1:
        try:
            user_id = int(sys.argv[1])
        except ValueError:
            print("Error: User ID must be an integer")
            sys.exit(1)
    else:
        try:
            user_id = int(input("Enter the User ID to delete: "))
        except ValueError:
            print("Error: User ID must be an integer")
            sys.exit(1)
    
    print(f"\n🔍 Looking up user with ID: {user_id}")
    
    # Check if user exists
    user = check_user_exists(user_id)
    if not user:
        print(f"❌ User with ID {user_id} not found in database")
        sys.exit(1)
    
    print(f"📋 Found user: {user['name']} ({user['email']})")
    
    # Count records
    print("\n📊 Counting associated records...")
    counts = count_user_records(user_id)
    
    print(f"  • Meal plans: {counts['meal_plans']}")
    print(f"  • Workout plans: {counts['workout_plans']}")
    print(f"  • Grocery lists: {counts['grocery_lists']}")
    
    total_records = sum(counts.values()) + 1  # +1 for user record
    print(f"\n📝 Total records to delete: {total_records}")
    
    # Confirmation
    print(f"\n⚠️  WARNING: This will permanently delete:")
    print(f"   - User: {user['name']} ({user['email']})")
    print(f"   - {counts['meal_plans']} meal plan(s)")
    print(f"   - {counts['workout_plans']} workout plan(s)")
    print(f"   - {counts['grocery_lists']} grocery list(s)")
    print(f"\n   This action CANNOT be undone!")
    
    confirm = input("\nType 'DELETE' to confirm deletion: ")
    
    if confirm != 'DELETE':
        print("❌ Deletion cancelled")
        sys.exit(0)
    
    # Perform deletion
    print(f"\n🗑️  Deleting user {user_id} and all associated data...")
    success = delete_user_data(user_id)
    
    if success:
        print(f"\n🎉 User {user_id} ({user['name']}) has been completely removed from the database")
    else:
        print(f"\n💥 Failed to delete user {user_id}")

if __name__ == "__main__":
    main()