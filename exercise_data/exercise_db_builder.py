import sqlite3
import json
import sys
from pathlib import Path

class ExerciseDatabaseBuilder:
    """
    Builds a relational SQLite database from exercises.json
    
    Schema:
    - exercises: main table with JSON arrays for instructions/images
    - muscles: lookup table for muscle names
    - exercise_primary_muscles: junction table
    - exercise_secondary_muscles: junction table
    """
    
    def __init__(self, json_file='exercises.json', db_file='exercises.db'):
        self.json_file = json_file
        self.db_file = db_file
        self.conn = None
        self.stats = {
            'exercises': 0,
            'muscles': 0,
            'primary_links': 0,
            'secondary_links': 0,
            'skipped': 0,
            'errors': []
        }
    
    def create_schema(self):
        """Create database tables with indexes"""
        cursor = self.conn.cursor()
        
        # Exercises table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS exercises (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                force TEXT,
                level TEXT NOT NULL,
                mechanic TEXT,
                equipment TEXT,
                category TEXT NOT NULL,
                instructions TEXT NOT NULL,
                images TEXT NOT NULL
            )
        ''')
        
        # Muscles lookup table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS muscles (
                muscle_id INTEGER PRIMARY KEY AUTOINCREMENT,
                muscle_name TEXT UNIQUE NOT NULL
            )
        ''')
        
        # Junction table for primary muscles
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS exercise_primary_muscles (
                exercise_id TEXT NOT NULL,
                muscle_id INTEGER NOT NULL,
                FOREIGN KEY (exercise_id) REFERENCES exercises(id),
                FOREIGN KEY (muscle_id) REFERENCES muscles(muscle_id),
                PRIMARY KEY (exercise_id, muscle_id)
            )
        ''')
        
        # Junction table for secondary muscles
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS exercise_secondary_muscles (
                exercise_id TEXT NOT NULL,
                muscle_id INTEGER NOT NULL,
                FOREIGN KEY (exercise_id) REFERENCES exercises(id),
                FOREIGN KEY (muscle_id) REFERENCES muscles(muscle_id),
                PRIMARY KEY (exercise_id, muscle_id)
            )
        ''')
        
        # Create indexes for common queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_level ON exercises(level)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_equipment ON exercises(equipment)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_category ON exercises(category)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_muscle_name ON muscles(muscle_name)')
        
        self.conn.commit()
        print("✓ Database schema created")
    
    def get_or_create_muscle(self, muscle_name):
        """Get muscle_id for a muscle name, creating it if needed"""
        cursor = self.conn.cursor()
        
        # Try to get existing
        cursor.execute('SELECT muscle_id FROM muscles WHERE muscle_name = ?', (muscle_name,))
        row = cursor.fetchone()
        
        if row:
            return row[0]
        
        # Create new
        cursor.execute('INSERT INTO muscles (muscle_name) VALUES (?)', (muscle_name,))
        self.stats['muscles'] += 1
        return cursor.lastrowid
    
    def insert_exercise(self, exercise):
        """Insert a single exercise and its relationships"""
        cursor = self.conn.cursor()
        
        # Validate required fields
        if not exercise.get('id') or not exercise.get('name'):
            self.stats['errors'].append(f"Missing id or name: {exercise.get('name', 'unknown')}")
            self.stats['skipped'] += 1
            return
        
        exercise_id = exercise['id']
        
        # Check if exercise already exists (skip duplicates)
        cursor.execute('SELECT id FROM exercises WHERE id = ?', (exercise_id,))
        if cursor.fetchone():
            self.stats['skipped'] += 1
            return
        
        # Convert arrays to JSON strings
        instructions_json = json.dumps(exercise.get('instructions', []))
        images_json = json.dumps(exercise.get('images', []))
        
        # Insert exercise
        try:
            cursor.execute('''
                INSERT INTO exercises (id, name, force, level, mechanic, equipment, category, instructions, images)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                exercise_id,
                exercise['name'],
                exercise.get('force'),  # Can be NULL
                exercise['level'],
                exercise.get('mechanic'),  # Can be NULL
                exercise.get('equipment'),  # Can be NULL
                exercise['category'],
                instructions_json,
                images_json
            ))
            self.stats['exercises'] += 1
        except sqlite3.Error as e:
            self.stats['errors'].append(f"Error inserting {exercise_id}: {str(e)}")
            self.stats['skipped'] += 1
            return
        
        # Insert primary muscles relationships
        for muscle in exercise.get('primaryMuscles', []):
            muscle_id = self.get_or_create_muscle(muscle)
            try:
                cursor.execute('''
                    INSERT INTO exercise_primary_muscles (exercise_id, muscle_id)
                    VALUES (?, ?)
                ''', (exercise_id, muscle_id))
                self.stats['primary_links'] += 1
            except sqlite3.IntegrityError:
                pass  # Duplicate, skip
        
        # Insert secondary muscles relationships
        for muscle in exercise.get('secondaryMuscles', []):
            muscle_id = self.get_or_create_muscle(muscle)
            try:
                cursor.execute('''
                    INSERT INTO exercise_secondary_muscles (exercise_id, muscle_id)
                    VALUES (?, ?)
                ''', (exercise_id, muscle_id))
                self.stats['secondary_links'] += 1
            except sqlite3.IntegrityError:
                pass  # Duplicate, skip
    
    def load_and_process(self):
        """Load JSON file and process all exercises"""
        # Check if file exists
        if not Path(self.json_file).exists():
            print(f"❌ Error: {self.json_file} not found!")
            return False
        
        print(f"📖 Reading {self.json_file}...")
        
        try:
            with open(self.json_file, 'r', encoding='utf-8') as f:
                exercises = json.load(f)
        except json.JSONDecodeError as e:
            print(f"❌ Error: Invalid JSON file - {str(e)}")
            return False
        
        if not isinstance(exercises, list):
            print("❌ Error: JSON file should contain an array of exercises")
            return False
        
        print(f"✓ Loaded {len(exercises)} exercises")
        print("💾 Inserting data...")
        
        # Process in batches for better performance
        batch_size = 100
        for i, exercise in enumerate(exercises):
            self.insert_exercise(exercise)
            
            # Commit every batch
            if (i + 1) % batch_size == 0:
                self.conn.commit()
                print(f"  Processed {i + 1}/{len(exercises)}...")
        
        # Final commit
        self.conn.commit()
        return True
    
    def print_statistics(self):
        """Print import statistics"""
        print("\n" + "="*50)
        print("📊 IMPORT STATISTICS")
        print("="*50)
        print(f"✓ Exercises inserted:        {self.stats['exercises']}")
        print(f"✓ Unique muscles:            {self.stats['muscles']}")
        print(f"✓ Primary muscle links:      {self.stats['primary_links']}")
        print(f"✓ Secondary muscle links:    {self.stats['secondary_links']}")
        print(f"⚠ Skipped/duplicates:        {self.stats['skipped']}")
        
        if self.stats['errors']:
            print(f"\n❌ Errors ({len(self.stats['errors'])}):")
            for error in self.stats['errors'][:10]:  # Show first 10
                print(f"  - {error}")
            if len(self.stats['errors']) > 10:
                print(f"  ... and {len(self.stats['errors']) - 10} more")
        
        print("="*50)
    
    def verify_database(self):
        """Quick verification queries"""
        cursor = self.conn.cursor()
        
        print("\n🔍 DATABASE VERIFICATION")
        print("-" * 50)
        
        # Sample exercise
        cursor.execute('''
            SELECT id, name, level, equipment, category
            FROM exercises
            LIMIT 1
        ''')
        sample = cursor.fetchone()
        if sample:
            print(f"Sample exercise: {sample[1]} ({sample[0]})")
            print(f"  Level: {sample[2]}, Equipment: {sample[3]}, Category: {sample[4]}")
        
        # Muscle distribution
        cursor.execute('''
            SELECT m.muscle_name, COUNT(*) as count
            FROM exercise_primary_muscles epm
            JOIN muscles m ON epm.muscle_id = m.muscle_id
            GROUP BY m.muscle_name
            ORDER BY count DESC
            LIMIT 5
        ''')
        print("\nTop 5 primary muscles:")
        for muscle, count in cursor.fetchall():
            print(f"  {muscle}: {count} exercises")
        
        # Equipment distribution
        cursor.execute('''
            SELECT equipment, COUNT(*) as count
            FROM exercises
            WHERE equipment IS NOT NULL
            GROUP BY equipment
            ORDER BY count DESC
            LIMIT 5
        ''')
        print("\nTop 5 equipment types:")
        for equipment, count in cursor.fetchall():
            print(f"  {equipment}: {count} exercises")
    
    def build(self):
        """Main method to build the database"""
        try:
            # Remove existing database
            if Path(self.db_file).exists():
                Path(self.db_file).unlink()
                print(f"🗑️  Removed existing {self.db_file}")
            
            # Connect to database
            self.conn = sqlite3.connect(self.db_file)
            print(f"🔌 Connected to {self.db_file}")
            
            # Create schema
            self.create_schema()
            
            # Load and process data
            if not self.load_and_process():
                return False
            
            # Print statistics
            self.print_statistics()
            
            # Verify
            self.verify_database()
            
            print(f"\n✅ Database created successfully: {self.db_file}")
            return True
            
        except Exception as e:
            print(f"\n❌ Fatal error: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        
        finally:
            if self.conn:
                self.conn.close()

def main():
    """Entry point"""
    print("🏋️  Exercise Database Builder")
    print("="*50)
    
    # Allow custom file names from command line
    json_file = sys.argv[1] if len(sys.argv) > 1 else 'exercises.json'
    db_file = sys.argv[2] if len(sys.argv) > 2 else 'exercises.db'
    
    builder = ExerciseDatabaseBuilder(json_file, db_file)
    success = builder.build()
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
