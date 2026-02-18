"""
=============================================================================
 Student Performance Dataset — Data Exploration
=============================================================================
 This script:
   1. Loads the dataset using pandas
   2. Displays the first 10 rows
   3. Shows basic dataset information (columns, data types, missing values)
   4. Provides a beginner-friendly explanation of every column
   5. Identifies the target variable and explains why
=============================================================================
"""

import pandas as pd

# ─────────────────────────────────────────────
# 1. Load the dataset
# ─────────────────────────────────────────────
# The CSV uses semicolons (;) as the delimiter, which is common in European datasets.
df = pd.read_csv("data/student-mat.csv", sep=";")
print("✅ Dataset loaded successfully!\n")

# ─────────────────────────────────────────────
# 2. Display the first 10 rows
# ─────────────────────────────────────────────
print("=" * 80)
print("FIRST 10 ROWS OF THE DATASET")
print("=" * 80)
print(df.head(10).to_string())
print()

# ─────────────────────────────────────────────
# 3. Basic dataset information
# ─────────────────────────────────────────────
print("=" * 80)
print("DATASET SHAPE")
print("=" * 80)
print(f"  Rows    : {df.shape[0]}")
print(f"  Columns : {df.shape[1]}")
print()

print("=" * 80)
print("COLUMN DATA TYPES")
print("=" * 80)
print(df.dtypes.to_string())
print()

print("=" * 80)
print("MISSING VALUES PER COLUMN")
print("=" * 80)
missing = df.isnull().sum()
if missing.sum() == 0:
    print("  🎉 No missing values found in any column!")
else:
    print(missing[missing > 0].to_string())
print()

print("=" * 80)
print("DESCRIPTIVE STATISTICS (NUMERIC COLUMNS)")
print("=" * 80)
print(df.describe().to_string())
print()

# ─────────────────────────────────────────────
# 4. Beginner-friendly column explanations
# ─────────────────────────────────────────────
column_explanations = {
    "school":     "Student's school — 'GP' (Gabriel Pereira) or 'MS' (Mousinho da Silveira).",
    "sex":        "Student's sex — 'F' (Female) or 'M' (Male).",
    "age":        "Student's age (numeric, from 15 to 22).",
    "address":    "Home address type — 'U' (Urban) or 'R' (Rural).",
    "famsize":    "Family size — 'LE3' (≤ 3 members) or 'GT3' (> 3 members).",
    "Pstatus":    "Parent's cohabitation status — 'T' (living Together) or 'A' (Apart).",
    "Medu":       "Mother's education level (0-4): 0=none, 1=primary (4th grade), "
                  "2=5th-9th grade, 3=secondary, 4=higher education.",
    "Fedu":       "Father's education level (0-4): same scale as Medu.",
    "Mjob":       "Mother's job — 'teacher', 'health', 'services', 'at_home', or 'other'.",
    "Fjob":       "Father's job — same categories as Mjob.",
    "reason":     "Reason for choosing this school — 'home' (close to home), 'reputation', "
                  "'course' (preference), or 'other'.",
    "guardian":   "Student's guardian — 'mother', 'father', or 'other'.",
    "traveltime": "Home-to-school travel time (1-4): 1=<15 min, 2=15-30 min, "
                  "3=30-60 min, 4=>60 min.",
    "studytime":  "Weekly study time (1-4): 1=<2 hrs, 2=2-5 hrs, 3=5-10 hrs, 4=>10 hrs.",
    "failures":   "Number of past class failures (0-4; 4 means ≥ 4 failures).",
    "schoolsup":  "Extra educational support from school — 'yes' or 'no'.",
    "famsup":     "Family educational support — 'yes' or 'no'.",
    "paid":       "Extra paid classes in the subject — 'yes' or 'no'.",
    "activities": "Participates in extra-curricular activities — 'yes' or 'no'.",
    "nursery":    "Attended nursery school — 'yes' or 'no'.",
    "higher":     "Wants to pursue higher education — 'yes' or 'no'.",
    "internet":   "Has internet access at home — 'yes' or 'no'.",
    "romantic":   "In a romantic relationship — 'yes' or 'no'.",
    "famrel":     "Quality of family relationships (1-5): 1=very bad → 5=excellent.",
    "freetime":   "Free time after school (1-5): 1=very low → 5=very high.",
    "goout":      "Going out with friends (1-5): 1=very low → 5=very high.",
    "Dalc":       "Workday alcohol consumption (1-5): 1=very low → 5=very high.",
    "Walc":       "Weekend alcohol consumption (1-5): 1=very low → 5=very high.",
    "health":     "Current health status (1-5): 1=very bad → 5=very good.",
    "absences":   "Number of school absences (numeric, 0-93).",
    "G1":         "First period (term) grade (0-20).",
    "G2":         "Second period (term) grade (0-20).",
    "G3":         "Final grade (0-20). ⭐ THIS IS THE TARGET VARIABLE.",
}

print("=" * 80)
print("COLUMN EXPLANATIONS (BEGINNER-FRIENDLY)")
print("=" * 80)
for col in df.columns:
    explanation = column_explanations.get(col, "No description available.")
    print(f"  {col:12s} → {explanation}")
print()

# ─────────────────────────────────────────────
# 5. Target variable identification
# ─────────────────────────────────────────────
print("=" * 80)
print("TARGET VARIABLE")
print("=" * 80)
print("""
  Column : G3  (Final Grade)
  Range  : 0 – 20

  WHY G3?
  -------
  • G3 represents the student's FINAL GRADE, which is the ultimate measure of
    academic performance — exactly what we want to predict.

  • All the other columns (demographics, family background, study habits,
    social behaviour, etc.) are FEATURES that may influence this final grade.

  • G1 and G2 (first and second period grades) are intermediate grades.
    They are highly correlated with G3, so depending on your goal you might:
      – INCLUDE them  → if you want maximum prediction accuracy.
      – EXCLUDE them  → if you want to predict performance using only
                         non-grade factors (more realistic for early prediction).

  TASK TYPE : Regression (predicting a continuous numeric value 0-20),
              or Classification if you bin grades into categories
              (e.g. Pass ≥ 10 vs Fail < 10).
""")

print("=" * 80)
print("🏁  Data exploration complete!")
print("=" * 80)
