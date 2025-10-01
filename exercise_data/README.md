# Exercise Database Documentation

A relational SQLite database for managing exercise information with detailed instructions, muscle targeting, and equipment requirements.

## Database Overview

This database stores comprehensive exercise data including:
- Exercise details (name, level, equipment, category)
- Step-by-step instructions
- Primary and secondary muscle groups
- Exercise images
- Equipment requirements
- Force types and mechanics

**Database File:** `exercises.db`

---

## Table Schemas

### 1. `exercises` (Main Table)

Stores core exercise information.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `id` | TEXT | No | Unique exercise identifier (Primary Key) |
| `name` | TEXT | No | Exercise name (e.g., "3/4 Sit-Up") |
| `force` | TEXT | Yes | Force type: "pull", "push", "static", or NULL |
| `level` | TEXT | No | Difficulty: "beginner", "intermediate", or "expert" |
| `mechanic` | TEXT | Yes | Movement type: "isolation", "compound", or NULL |
| `equipment` | TEXT | Yes | Required equipment (see Equipment Types below) |
| `category` | TEXT | No | Exercise category (see Categories below) |
| `instructions` | TEXT | No | JSON array of step-by-step instructions |
| `images` | TEXT | No | JSON array of image file paths |

**Indexes:**
- `idx_level` on `level`
- `idx_equipment` on `equipment`
- `idx_category` on `category`

**JSON Array Format Examples:**
```json
instructions: ["Step 1 description", "Step 2 description", "Step 3 description"]
images: ["Exercise_Name/0.jpg", "Exercise_Name/1.jpg"]
```

---

### 2. `muscles` (Lookup Table)

Stores unique muscle group names.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `muscle_id` | INTEGER | No | Auto-incrementing ID (Primary Key) |
| `muscle_name` | TEXT | No | Muscle group name (Unique) |

**Index:**
- `idx_muscle_name` on `muscle_name`

**Valid Muscle Names:**
- abdominals
- abductors
- adductors
- biceps
- calves
- chest
- forearms
- glutes
- hamstrings
- lats
- lower back
- middle back
- neck
- quadriceps
- shoulders
- traps
- triceps

---

### 3. `exercise_primary_muscles` (Junction Table)

Links exercises to their primary target muscles (many-to-many).

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `exercise_id` | TEXT | No | Foreign key to `exercises.id` |
| `muscle_id` | INTEGER | No | Foreign key to `muscles.muscle_id` |

**Primary Key:** Composite (`exercise_id`, `muscle_id`)

---

### 4. `exercise_secondary_muscles` (Junction Table)

Links exercises to their secondary/supporting muscles (many-to-many).

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `exercise_id` | TEXT | No | Foreign key to `exercises.id` |
| `muscle_id` | INTEGER | No | Foreign key to `muscles.muscle_id` |

**Primary Key:** Composite (`exercise_id`, `muscle_id`)

---

## Enumerated Values

### Equipment Types
- `null` - No equipment needed
- `body only` - Bodyweight exercises
- `dumbbell`
- `barbell`
- `kettlebells`
- `cable`
- `machine`
- `bands`
- `medicine ball`
- `exercise ball`
- `foam roll`
- `e-z curl bar`
- `other`

### Categories
- `strength` - Strength training
- `stretching` - Flexibility and stretching
- `cardio` - Cardiovascular exercises
- `powerlifting` - Powerlifting movements
- `olympic weightlifting` - Olympic lifts
- `strongman` - Strongman events
- `plyometrics` - Jump/explosive training

### Force Types
- `push` - Pushing movements
- `pull` - Pulling movements
- `static` - Isometric/static holds
- `null` - Not applicable

### Difficulty Levels
- `beginner`
- `intermediate`
- `expert`

### Mechanic Types
- `compound` - Multi-joint movements
- `isolation` - Single-joint movements
- `null` - Not applicable

---

## Entity Relationships

```
exercises (1) ──→ (many) exercise_primary_muscles (many) ──→ (1) muscles
exercises (1) ──→ (many) exercise_secondary_muscles (many) ──→ (1) muscles
```

**Relationship Types:**
- An exercise can target **multiple primary muscles**
- An exercise can target **multiple secondary muscles**
- A muscle can be targeted by **multiple exercises**

---

## Common Query Examples

### 1. Get Exercise with Full Details

```sql
SELECT 
    e.id,
    e.name,
    e.level,
    e.equipment,
    e.category,
    e.instructions,
    e.images
FROM exercises e
WHERE e.id = '3_4_Sit-Up';
```

### 2. Get Exercise with Primary Muscles

```sql
SELECT 
    e.name,
    GROUP_CONCAT(m.muscle_name, ', ') as primary_muscles
FROM exercises e
LEFT JOIN exercise_primary_muscles epm ON e.id = epm.exercise_id
LEFT JOIN muscles m ON epm.muscle_id = m.muscle_id
WHERE e.id = 'Alternate_Hammer_Curl'
GROUP BY e.id, e.name;
```

### 3. Find Beginner Exercises by Equipment

```sql
SELECT 
    id,
    name,
    category
FROM exercises
WHERE level = 'beginner'
    AND equipment = 'dumbbell'
ORDER BY name;
```

### 4. Find Exercises Targeting Specific Muscle

```sql
SELECT 
    e.name,
    e.level,
    e.equipment
FROM exercises e
JOIN exercise_primary_muscles epm ON e.id = epm.exercise_id
JOIN muscles m ON epm.muscle_id = m.muscle_id
WHERE m.muscle_name = 'chest'
ORDER BY e.level, e.name;
```

### 5. Find Bodyweight Exercises for Beginners

```sql
SELECT 
    name,
    category,
    instructions
FROM exercises
WHERE level = 'beginner'
    AND (equipment = 'body only' OR equipment IS NULL)
ORDER BY category, name;
```

### 6. Get Exercise Count by Category

```sql
SELECT 
    category,
    COUNT(*) as exercise_count
FROM exercises
GROUP BY category
ORDER BY exercise_count DESC;
```

### 7. Find Push Exercises for Intermediate Level

```sql
SELECT 
    name,
    equipment,
    mechanic
FROM exercises
WHERE force = 'push'
    AND level = 'intermediate'
ORDER BY equipment, name;
```

### 8. Get Full Exercise Information (with muscles)

```sql
SELECT 
    e.id,
    e.name,
    e.level,
    e.equipment,
    e.category,
    e.force,
    e.mechanic,
    GROUP_CONCAT(DISTINCT pm.muscle_name) as primary_muscles,
    GROUP_CONCAT(DISTINCT sm.muscle_name) as secondary_muscles,
    e.instructions,
    e.images
FROM exercises e
LEFT JOIN exercise_primary_muscles epm ON e.id = epm.exercise_id
LEFT JOIN muscles pm ON epm.muscle_id = pm.muscle_id
LEFT JOIN exercise_secondary_muscles esm ON e.id = esm.exercise_id
LEFT JOIN muscles sm ON esm.muscle_id = sm.muscle_id
WHERE e.name LIKE '%Curl%'
GROUP BY e.id;
```

---

## Parsing JSON Arrays in Applications

### JavaScript Example

```javascript
// Parse instructions
const row = db.query("SELECT instructions FROM exercises WHERE id = ?", [exerciseId]);
const instructions = JSON.parse(row.instructions);

// Now you have: ["Step 1", "Step 2", "Step 3"]
instructions.forEach((step, index) => {
    console.log(`${index + 1}. ${step}`);
});
```

### Python Example

```python
import json
import sqlite3

conn = sqlite3.connect('exercises.db')
cursor = conn.cursor()

cursor.execute("SELECT instructions, images FROM exercises WHERE id = ?", ('Air_Bike',))
row = cursor.fetchone()

instructions = json.loads(row[0])  # Parse JSON array
images = json.loads(row[1])        # Parse JSON array

for i, step in enumerate(instructions, 1):
    print(f"{i}. {step}")
```

---

## Database Statistics

After importing a typical dataset, you can expect:
- **~800-1000 exercises**
- **17 unique muscle groups**
- **~850 primary muscle relationships**
- **~400-500 secondary muscle relationships**

---

## Building the Database

To create or rebuild the database from `exercises.json`:

```bash
python exercise_db_builder.py
```

This will:
1. Create the schema with all tables and indexes
2. Import all exercises from the JSON file
3. Auto-populate the muscles table
4. Create all muscle relationships
5. Display import statistics

---

## Data Integrity Rules

1. **Primary Keys:** Exercise IDs must be unique
2. **Foreign Keys:** All muscle relationships reference valid exercises and muscles
3. **Required Fields:** id, name, level, category, instructions, images
4. **Optional Fields:** force, mechanic, equipment (can be NULL)
5. **Duplicate Handling:** Duplicate exercise IDs are skipped during import

---

## Notes

- **Image Paths:** Stored as relative paths (e.g., `"Exercise_Name/0.jpg"`)
- **NULL vs Empty:** Empty arrays `[]` are stored as JSON; NULL fields indicate optional/missing data
- **Case Sensitivity:** SQLite is case-insensitive for string comparisons by default
- **Muscle Names:** Always lowercase in the database

---

## Version

- **Database Version:** 1.0
- **Schema Date:** 2025-01-30
- **Compatible with:** SQLite 3.x

---

## Support

For issues or questions about the database structure, refer to:
- `schema.json` - Original JSON schema definition
- `exercise_db_builder.py` - Database builder script