<artifact identifier="readme-exercise-db" type="text/markdown" title="README.md - Exercise Database Documentation">
# Exercise Database Documentation

A relational SQLite database for managing exercise information with detailed instructions, muscle targeting, equipment requirements, contraindications, modifications, and programming recommendations.

## Database Overview

This database stores comprehensive exercise data including:
- Exercise details (name, level, equipment, category)
- Step-by-step instructions
- Primary and secondary muscle groups
- Exercise images
- Equipment requirements
- Force types and mechanics
- **Contraindications and injury considerations**
- **Exercise modifications (easier, harder, injury-friendly)**
- **Safety notes, form cues, and common mistakes**
- **Programming recommendations (sets, reps, rest periods)**

**Database File:** `exercises.db`

---

## Table Schemas

### Core Exercise Tables

#### 1. `exercises` (Main Table)

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

#### 2. `muscles` (Lookup Table)

Stores unique muscle group names.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `muscle_id` | INTEGER | No | Auto-incrementing ID (Primary Key) |
| `muscle_name` | TEXT | No | Muscle group name (Unique) |

**Index:**
- `idx_muscle_name` on `muscle_name`

**Valid Muscle Names:**
- abdominals, abductors, adductors, biceps, calves, chest, forearms, glutes, hamstrings, lats, lower back, middle back, neck, quadriceps, shoulders, traps, triceps

---

#### 3. `exercise_primary_muscles` (Junction Table)

Links exercises to their primary target muscles (many-to-many).

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `exercise_id` | TEXT | No | Foreign key to `exercises.id` |
| `muscle_id` | INTEGER | No | Foreign key to `muscles.muscle_id` |

**Primary Key:** Composite (`exercise_id`, `muscle_id`)

---

#### 4. `exercise_secondary_muscles` (Junction Table)

Links exercises to their secondary/supporting muscles (many-to-many).

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `exercise_id` | TEXT | No | Foreign key to `exercises.id` |
| `muscle_id` | INTEGER | No | Foreign key to `muscles.muscle_id` |

**Primary Key:** Composite (`exercise_id`, `muscle_id`)

---

### Contraindication & Modification Tables

#### 5. `modification_categories` (Lookup Table)

Stores categories for contraindications and difficulty modifications.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `category_id` | INTEGER | No | Auto-incrementing ID (Primary Key) |
| `category_name` | TEXT | No | Category name (Unique) |
| `category_type` | TEXT | No | Type: "contraindication" or "difficulty" |
| `display_order` | INTEGER | Yes | Sort order for display |

**Indexes:**
- `idx_mod_cat_name` on `category_name`
- `idx_mod_cat_type` on `category_type`

**Contraindication Categories:**
- back and spinal issues
- knee and foot issues
- chest and shoulder issues
- hip or lumbar issues
- arm and hand issues
- Pregnancy
- chronical or neurological issues

**Difficulty Categories:**
- easier
- harder

---

#### 6. `contraindications` (Lookup Table)

Stores specific contraindication names with severity ratings.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `contraindication_id` | INTEGER | No | Auto-incrementing ID (Primary Key) |
| `contraindication_name` | TEXT | No | Contraindication name (Unique) |
| `category_id` | INTEGER | No | Foreign key to `modification_categories.category_id` |
| `severity` | TEXT | Yes | Severity level: "low", "moderate", or "high" |

**Indexes:**
- `idx_contra_name` on `contraindication_name`
- `idx_contra_category` on `category_id`

---

#### 7. `exercise_contraindications` (Junction Table)

Links exercises to their contraindications with specific reasons.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `exercise_id` | TEXT | No | Foreign key to `exercises.id` |
| `contraindication_id` | INTEGER | No | Foreign key to `contraindications.contraindication_id` |
| `specific_reason` | TEXT | Yes | Detailed explanation for this contraindication |

**Primary Key:** Composite (`exercise_id`, `contraindication_id`)

**Index:**
- `idx_ex_contra_exercise` on `exercise_id`

---

#### 8. `exercise_modifications`

Stores exercise modifications for different needs (easier, harder, injury-specific).

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `modification_id` | INTEGER | No | Auto-incrementing ID (Primary Key) |
| `exercise_id` | TEXT | No | Foreign key to `exercises.id` |
| `category_id` | INTEGER | No | Foreign key to `modification_categories.category_id` |
| `modification_text` | TEXT | No | Description of the modification |

**Unique Constraint:** (`exercise_id`, `category_id`) - One modification per category per exercise

**Indexes:**
- `idx_mod_exercise` on `exercise_id`
- `idx_mod_category` on `category_id`

---

### Safety & Form Tables

#### 9. `exercise_safety_notes`

Stores safety warnings and precautions for exercises.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `note_id` | INTEGER | No | Auto-incrementing ID (Primary Key) |
| `exercise_id` | TEXT | No | Foreign key to `exercises.id` |
| `note_text` | TEXT | No | Safety note content |
| `display_order` | INTEGER | Yes | Sort order for display |

---

#### 10. `exercise_form_cues`

Stores form cues and technique tips for proper execution.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `cue_id` | INTEGER | No | Auto-incrementing ID (Primary Key) |
| `exercise_id` | TEXT | No | Foreign key to `exercises.id` |
| `cue_text` | TEXT | No | Form cue content |
| `display_order` | INTEGER | Yes | Sort order for display |

---

#### 11. `exercise_common_mistakes`

Stores common mistakes to avoid when performing exercises.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `mistake_id` | INTEGER | No | Auto-incrementing ID (Primary Key) |
| `exercise_id` | TEXT | No | Foreign key to `exercises.id` |
| `mistake_text` | TEXT | No | Common mistake description |
| `display_order` | INTEGER | Yes | Sort order for display |

---

### Programming Tables

#### 12. `exercise_programming`

Stores programming recommendations for sets, reps, rest periods, and time estimates.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `exercise_id` | TEXT | No | Foreign key to `exercises.id` (Primary Key) |
| `sets_beginner` | INTEGER | Yes | Recommended sets for beginners |
| `sets_intermediate` | INTEGER | Yes | Recommended sets for intermediate |
| `sets_advanced` | INTEGER | Yes | Recommended sets for advanced |
| `reps_strength` | INTEGER | Yes | Reps for strength goals |
| `reps_hypertrophy` | INTEGER | Yes | Reps for muscle growth |
| `reps_endurance` | INTEGER | Yes | Reps for endurance |
| `rest_beginner` | INTEGER | Yes | Rest period (seconds) for beginners |
| `rest_intermediate` | INTEGER | Yes | Rest period (seconds) for intermediate |
| `rest_advanced` | INTEGER | Yes | Rest period (seconds) for advanced |
| `calories_beginner` | REAL | Yes | Estimated calories per minute (beginner) |
| `calories_intermediate` | REAL | Yes | Estimated calories per minute (intermediate) |
| `calories_advanced` | REAL | Yes | Estimated calories per minute (advanced) |
| `time_beginner` | INTEGER | Yes | Estimated time (minutes) for beginners |
| `time_intermediate` | INTEGER | Yes | Estimated time (minutes) for intermediate |
| `time_advanced` | INTEGER | Yes | Estimated time (minutes) for advanced |

---

## Enumerated Values

### Equipment Types
- `null` - No equipment needed
- `body only` - Bodyweight exercises
- `dumbbell`, `barbell`, `kettlebells`, `cable`, `machine`, `bands`, `medicine ball`, `exercise ball`, `foam roll`, `e-z curl bar`, `other`

### Categories
- `strength`, `stretching`, `cardio`, `powerlifting`, `olympic weightlifting`, `strongman`, `plyometrics`

### Force Types
- `push`, `pull`, `static`, `null`

### Difficulty Levels
- `beginner`, `intermediate`, `expert`

### Mechanic Types
- `compound`, `isolation`, `null`

### Severity Levels (Contraindications)
- `low` - Minor concern, proceed with caution
- `moderate` - Significant concern, modification recommended
- `high` - High risk, avoid or consult professional

---

## Entity Relationships

```
exercises (1) ──→ (many) exercise_primary_muscles (many) ──→ (1) muscles
exercises (1) ──→ (many) exercise_secondary_muscles (many) ──→ (1) muscles
exercises (1) ──→ (many) exercise_contraindications (many) ──→ (1) contraindications
contraindications (many) ──→ (1) modification_categories
exercises (1) ──→ (many) exercise_modifications (many) ──→ (1) modification_categories
exercises (1) ──→ (many) exercise_safety_notes
exercises (1) ──→ (many) exercise_form_cues
exercises (1) ──→ (many) exercise_common_mistakes
exercises (1) ──→ (0..1) exercise_programming
```

**Relationship Types:**
- An exercise can have **multiple contraindications**
- An exercise can have **multiple modifications** (one per category)
- An exercise can have **multiple safety notes, form cues, and common mistakes**
- An exercise can have **one set of programming recommendations**

---

## Common Query Examples

### Basic Exercise Queries

#### 1. Get Exercise with Full Details

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

#### 2. Get Exercise with Primary Muscles

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

#### 3. Find Beginner Exercises by Equipment

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

---

### Contraindication & Modification Queries

#### 4. Get All Contraindications for an Exercise

```sql
SELECT 
    e.name,
    c.contraindication_name,
    c.severity,
    mc.category_name,
    ec.specific_reason
FROM exercises e
JOIN exercise_contraindications ec ON e.id = ec.exercise_id
JOIN contraindications c ON ec.contraindication_id = c.contraindication_id
JOIN modification_categories mc ON c.category_id = mc.category_id
WHERE e.id = 'Barbell_Squat'
ORDER BY c.severity DESC;
```

#### 5. Find Exercises Safe for Specific Condition

```sql
-- Find exercises that DON'T have "back pain" contraindication
SELECT DISTINCT e.id, e.name, e.level
FROM exercises e
WHERE e.id NOT IN (
    SELECT ec.exercise_id
    FROM exercise_contraindications ec
    JOIN contraindications c ON ec.contraindication_id = c.contraindication_id
    WHERE c.contraindication_name LIKE '%back%'
)
AND e.level = 'beginner'
ORDER BY e.name;
```

#### 6. Get Exercise Modifications

```sql
SELECT 
    e.name,
    mc.category_name,
    mc.category_type,
    em.modification_text
FROM exercises e
JOIN exercise_modifications em ON e.id = em.exercise_id
JOIN modification_categories mc ON em.category_id = mc.category_id
WHERE e.id = 'Push-Ups'
ORDER BY mc.category_type, mc.category_name;
```

#### 7. Find Easier Alternatives for an Exercise

```sql
SELECT 
    e.name AS original_exercise,
    em.modification_text AS easier_version
FROM exercises e
JOIN exercise_modifications em ON e.id = em.exercise_id
JOIN modification_categories mc ON em.category_id = mc.category_id
WHERE e.name LIKE '%Squat%'
    AND mc.category_name = 'easier';
```

---

### Safety & Form Queries

#### 8. Get Safety Notes and Form Cues for Exercise

```sql
SELECT 
    e.name,
    'Safety' as type,
    sn.note_text as content,
    sn.display_order
FROM exercises e
JOIN exercise_safety_notes sn ON e.id = sn.exercise_id
WHERE e.id = 'Deadlift'

UNION ALL

SELECT 
    e.name,
    'Form Cue' as type,
    fc.cue_text as content,
    fc.display_order
FROM exercises e
JOIN exercise_form_cues fc ON e.id = fc.exercise_id
WHERE e.id = 'Deadlift'

ORDER BY type, display_order;
```

#### 9. Get Common Mistakes for Exercise

```sql
SELECT 
    e.name,
    cm.mistake_text,
    cm.display_order
FROM exercises e
JOIN exercise_common_mistakes cm ON e.id = cm.exercise_id
WHERE e.id = 'Barbell_Bench_Press'
ORDER BY cm.display_order;
```

---

### Programming Queries

#### 10. Get Programming Recommendations

```sql
SELECT 
    e.name,
    e.level,
    p.sets_beginner,
    p.sets_intermediate,
    p.sets_advanced,
    p.reps_strength,
    p.reps_hypertrophy,
    p.rest_beginner,
    p.rest_intermediate,
    p.rest_advanced
FROM exercises e
JOIN exercise_programming p ON e.id = p.exercise_id
WHERE e.category = 'strength'
    AND e.equipment = 'barbell'
LIMIT 10;
```

#### 11. Calculate Workout Duration

```sql
SELECT 
    e.name,
    e.level,
    p.time_beginner || ' minutes' as duration,
    p.calories_beginner || ' cal/min' as calorie_burn
FROM exercises e
JOIN exercise_programming p ON e.id = p.exercise_id
WHERE e.level = 'beginner'
    AND p.time_beginner IS NOT NULL
ORDER BY p.time_beginner;
```

---

### Comprehensive Queries

#### 12. Get Complete Exercise Profile

```sql
SELECT 
    e.id,
    e.name,
    e.level,
    e.equipment,
    e.category,
    GROUP_CONCAT(DISTINCT pm.muscle_name) as primary_muscles,
    GROUP_CONCAT(DISTINCT sm.muscle_name) as secondary_muscles,
    (SELECT GROUP_CONCAT(c.contraindication_name, '; ')
     FROM exercise_contraindications ec
     JOIN contraindications c ON ec.contraindication_id = c.contraindication_id
     WHERE ec.exercise_id = e.id) as contraindications,
    p.sets_intermediate,
    p.reps_hypertrophy,
    p.rest_intermediate,
    e.instructions
FROM exercises e
LEFT JOIN exercise_primary_muscles epm ON e.id = epm.exercise_id
LEFT JOIN muscles pm ON epm.muscle_id = pm.muscle_id
LEFT JOIN exercise_secondary_muscles esm ON e.id = esm.exercise_id
LEFT JOIN muscles sm ON esm.muscle_id = sm.muscle_id
LEFT JOIN exercise_programming p ON e.id = p.exercise_id
WHERE e.name LIKE '%Press%'
GROUP BY e.id
LIMIT 5;
```

#### 13. Build Safe Workout for User with Constraints

```sql
-- Find beginner chest exercises safe for someone with shoulder issues
SELECT DISTINCT
    e.name,
    e.equipment,
    GROUP_CONCAT(DISTINCT pm.muscle_name) as muscles,
    em.modification_text as modification
FROM exercises e
JOIN exercise_primary_muscles epm ON e.id = epm.exercise_id
JOIN muscles pm ON epm.muscle_id = pm.muscle_id
LEFT JOIN exercise_modifications em ON e.id = em.exercise_id
LEFT JOIN modification_categories mc ON em.category_id = mc.category_id
WHERE pm.muscle_name = 'chest'
    AND e.level = 'beginner'
    AND e.id NOT IN (
        SELECT ec.exercise_id
        FROM exercise_contraindications ec
        JOIN contraindications c ON ec.contraindication_id = c.contraindication_id
        WHERE c.contraindication_name LIKE '%shoulder%'
    )
    AND (mc.category_name = 'chest and shoulder issues' OR mc.category_name IS NULL)
GROUP BY e.id
ORDER BY e.name;
```

---

## Database Statistics

After importing a typical dataset, you can expect:
- **~600-900 exercises** (base exercises)
- **17 unique muscle groups**
- **~850 primary muscle relationships**
- **~400-500 secondary muscle relationships**
- **10 modification categories**
- **~200-300 unique contraindications**
- **~200+ exercise-contraindication links**
- **~2,400+ exercise modifications**
- **Multiple safety notes, form cues, and mistakes per exercise**
- **Programming data for most exercises**

---

## Building the Database

### Initial Database Creation

To create the base database from `exercises.json`:

```bash
python exercise_db_builder.py
```

### Enrichment Migration

To add contraindications, modifications, safety notes, and programming data:

```bash
python database_enricher_v2.py
```

This will:
1. Create enrichment schema with all new tables and indexes
2. Populate modification categories
3. Import contraindications and create relationships
4. Add exercise modifications (easier, harder, injury-friendly)
5. Import safety notes, form cues, and common mistakes
6. Add programming recommendations (sets, reps, rest periods)
7. Display migration statistics

---

## Data Integrity Rules

1. **Primary Keys:** Exercise IDs must be unique
2. **Foreign Keys:** All relationships reference valid exercises, muscles, categories, and contraindications
3. **Required Fields:** 
   - `exercises`: id, name, level, category, instructions, images
   - `contraindications`: contraindication_name, category_id
   - `exercise_modifications`: exercise_id, category_id, modification_text
4. **Optional Fields:** force, mechanic, equipment, severity, specific_reason, programming values
5. **Unique Constraints:** 
   - One modification per category per exercise
   - Unique contraindication names
   - Unique muscle names
6. **Check Constraints:**
   - category_type IN ('contraindication', 'difficulty')
   - severity IN ('low', 'moderate', 'high')

---

## Notes

- **Image Paths:** Stored as relative paths (e.g., `"Exercise_Name/0.jpg"`)
- **NULL vs Empty:** Empty arrays `[]` are stored as JSON; NULL fields indicate optional/missing data
- **Case Sensitivity:** SQLite is case-insensitive for string comparisons by default
- **Muscle Names:** Always lowercase in the database
- **Display Order:** Use `display_order` fields to maintain consistent UI ordering
- **Text Fields:** Safety notes, form cues, and mistakes are stored as plain text (not JSON)
- **Programming Flexibility:** All programming fields are nullable to accommodate exercises without specific recommendations

---

## Version

- **Database Version:** 2.0
- **Schema Date:** 2025-10-06
- **Compatible with:** SQLite 3.x
- **Major Changes in v2.0:**
  - Added contraindication system with categories
  - Added exercise modifications (easier, harder, injury-specific)
  - Added safety notes, form cues, and common mistakes tables
  - Added comprehensive programming recommendations

---

## Support

For issues or questions about the database structure, refer to:
- `schema.json` - Original JSON schema definition
- `exercise_db_builder.py` - Base database builder script
- `database_enricher_v2.py` - Enrichment migration script
- `exercise_enriched_corrected.json` - Enriched exercise data source
</artifact>
