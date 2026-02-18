"""
=============================================================================
 Student Performance Dataset — Feature Engineering & Encoding
=============================================================================
 This script:
   1. Loads the dataset and separates the target (G3) from features
   2. Creates TWO feature sets:
      a) Feature Set A — includes G1 and G2  (all available info)
      b) Feature Set B — excludes G1 and G2  (early-prediction scenario)
   3. Encodes all categorical features so ML models can use them
   4. Prints clear, beginner-friendly explanations at every step
=============================================================================
"""

import pandas as pd

# ─────────────────────────────────────────────
# STEP 1 · Load the dataset
# ─────────────────────────────────────────────
df = pd.read_csv("data/student-mat.csv", sep=";")
print("✅ Dataset loaded successfully!")
print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")

# ─────────────────────────────────────────────
# STEP 2 · Separate Target (y) and Features (X)
# ─────────────────────────────────────────────
# WHAT IS A TARGET VARIABLE?
#   The target (also called the "label") is the column we want to PREDICT.
#   Here it is G3 — the student's final grade (0-20).
#
# WHAT ARE FEATURES?
#   Features (also called "inputs" or "predictors") are ALL the other columns.
#   They describe the student and are used by the model to make predictions.

y = df["G3"]                   # Target: final grade
X = df.drop(columns=["G3"])    # Features: everything EXCEPT the final grade

print("=" * 80)
print("STEP 2 — SEPARATE TARGET (y) AND FEATURES (X)")
print("=" * 80)
print(f"  Target column  : G3  (final grade)")
print(f"  Target shape   : {y.shape}")
print(f"  Features shape : {X.shape}  (G3 removed)")
print()

# ─────────────────────────────────────────────
# STEP 3 · Create two feature sets
# ─────────────────────────────────────────────
# WHY TWO SETS?
#   G1 and G2 are the student's grades from earlier terms. They are very
#   strongly correlated with the final grade G3 (because a student who
#   scores high early usually scores high at the end too).
#
#   • Feature Set A (WITH G1 & G2):
#       – Gives the model more information → higher accuracy.
#       – Use this when you ALREADY HAVE the first two term grades.
#
#   • Feature Set B (WITHOUT G1 & G2):
#       – Forces the model to predict G3 from demographics, family,
#         study habits, and social factors ONLY.
#       – More realistic for EARLY PREDICTION (e.g. at the start of
#         the school year, before any exams have been taken).
#       – Helps you understand which non-grade factors matter most.

X_with_grades    = X.copy()                           # Set A: keep G1, G2
X_without_grades = X.drop(columns=["G1", "G2"])       # Set B: drop G1, G2

print("=" * 80)
print("STEP 3 — TWO FEATURE SETS")
print("=" * 80)
print(f"  Feature Set A (with G1 & G2)    : {X_with_grades.shape}")
print(f"  Feature Set B (without G1 & G2) : {X_without_grades.shape}")
print()

# ─────────────────────────────────────────────
# STEP 4 · Identify categorical vs numeric columns
# ─────────────────────────────────────────────
# WHY DOES THIS MATTER?
#   Machine learning models work with NUMBERS. Columns that contain text
#   (like "GP", "F", "yes") are called "categorical" — they represent
#   categories, not quantities. We need to convert (encode) them into
#   numbers before feeding them into any ML algorithm.

# Categorical columns are those with dtype 'object' (text/string).
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols     = X.select_dtypes(include=["number"]).columns.tolist()

print("=" * 80)
print("STEP 4 — IDENTIFY COLUMN TYPES")
print("=" * 80)
print(f"  Categorical columns ({len(categorical_cols)}):")
for col in categorical_cols:
    unique_vals = df[col].unique()
    print(f"    • {col:12s} → {len(unique_vals)} unique values: {list(unique_vals)}")
print()
print(f"  Numeric columns ({len(numeric_cols)}):")
print(f"    {numeric_cols}")
print()

# ─────────────────────────────────────────────
# STEP 5 · Encode categorical features
# ─────────────────────────────────────────────
# WHAT IS ENCODING?
#   Encoding converts text labels into numbers so that ML algorithms can
#   process them. We use TWO common strategies:
#
#   1. LABEL ENCODING (binary columns only)
#      Columns with exactly 2 values (like yes/no, M/F, U/R) are mapped
#      to 0 and 1. This is simple and doesn't add extra columns.
#
#   2. ONE-HOT ENCODING (multi-category columns)
#      Columns with 3+ values (like Mjob: teacher/health/services/at_home/other)
#      are split into multiple binary (0/1) columns — one per category.
#      This avoids implying a false ordering (e.g. "teacher" > "health").
#
# WHY NOT JUST USE NUMBERS LIKE 1, 2, 3?
#   If you assign teacher=1, health=2, services=3, the model might think
#   services > health > teacher, which is meaningless for categories.
#   One-hot encoding avoids this problem entirely.

# --- Identify binary vs multi-category columns ---
binary_cols = [col for col in categorical_cols if df[col].nunique() == 2]
multi_cols  = [col for col in categorical_cols if df[col].nunique() > 2]

print("=" * 80)
print("STEP 5 — ENCODE CATEGORICAL FEATURES")
print("=" * 80)
print(f"  Binary columns  (Label Encoding) : {binary_cols}")
print(f"  Multi-category  (One-Hot Encoding): {multi_cols}")
print()

# --- Define the binary mappings ---
# We manually define mappings so the output is transparent and reproducible.
binary_mappings = {
    "school":     {"GP": 0, "MS": 1},
    "sex":        {"F": 0, "M": 1},
    "address":    {"R": 0, "U": 1},
    "famsize":    {"LE3": 0, "GT3": 1},
    "Pstatus":    {"A": 0, "T": 1},
    "schoolsup":  {"no": 0, "yes": 1},
    "famsup":     {"no": 0, "yes": 1},
    "paid":       {"no": 0, "yes": 1},
    "activities": {"no": 0, "yes": 1},
    "nursery":    {"no": 0, "yes": 1},
    "higher":     {"no": 0, "yes": 1},
    "internet":   {"no": 0, "yes": 1},
    "romantic":   {"no": 0, "yes": 1},
}

def encode_features(X_input):
    """Apply label encoding to binary columns and one-hot encoding to multi-category columns."""
    X_encoded = X_input.copy()

    # 1. Label-encode binary columns
    for col, mapping in binary_mappings.items():
        if col in X_encoded.columns:
            X_encoded[col] = X_encoded[col].map(mapping)

    # 2. One-hot-encode multi-category columns
    #    pd.get_dummies automatically creates new binary columns and drops
    #    the original text column. drop_first=True removes one dummy column
    #    per feature to avoid multicollinearity (a statistical issue where
    #    columns are perfectly predictable from each other).
    X_encoded = pd.get_dummies(X_encoded, columns=multi_cols, drop_first=True)

    return X_encoded

# --- Apply encoding to both feature sets ---
X_A_encoded = encode_features(X_with_grades)
X_B_encoded = encode_features(X_without_grades)

print("-" * 80)
print("  BINARY ENCODING MAPPINGS APPLIED:")
print("-" * 80)
for col, mapping in binary_mappings.items():
    original_vals = list(mapping.keys())
    encoded_vals  = list(mapping.values())
    print(f"    {col:12s} :  {original_vals[0]} → {encoded_vals[0]},  "
          f"{original_vals[1]} → {encoded_vals[1]}")
print()

print("-" * 80)
print("  ONE-HOT ENCODED COLUMNS (multi-category):")
print("-" * 80)
for col in multi_cols:
    new_cols = [c for c in X_A_encoded.columns if c.startswith(col + "_")]
    print(f"    {col:12s} → {new_cols}")
print()

# ─────────────────────────────────────────────
# STEP 6 · Final summary
# ─────────────────────────────────────────────
print("=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
print(f"  Target (y)  : {y.shape[0]} values  |  dtype: {y.dtype}")
print()
print(f"  Feature Set A (with G1 & G2):")
print(f"    Before encoding : {X_with_grades.shape}")
print(f"    After encoding  : {X_A_encoded.shape}")
print()
print(f"  Feature Set B (without G1 & G2):")
print(f"    Before encoding : {X_without_grades.shape}")
print(f"    After encoding  : {X_B_encoded.shape}")
print()
print("  All columns are now NUMERIC — ready for machine learning! ✅")
print()

# --- Show a preview of the encoded data ---
print("-" * 80)
print("  PREVIEW — Feature Set A (first 5 rows, first 15 columns):")
print("-" * 80)
print(X_A_encoded.iloc[:5, :15].to_string())
print()
print("-" * 80)
print("  PREVIEW — Feature Set B (first 5 rows, first 15 columns):")
print("-" * 80)
print(X_B_encoded.iloc[:5, :15].to_string())
print()

# --- Confirm no text columns remain ---
remaining_object_A = X_A_encoded.select_dtypes(include=["object"]).columns.tolist()
remaining_object_B = X_B_encoded.select_dtypes(include=["object"]).columns.tolist()
if not remaining_object_A and not remaining_object_B:
    print("  ✅ Verification: ZERO text/object columns remain — encoding is complete.")
else:
    print(f"  ⚠️  Remaining text columns in Set A: {remaining_object_A}")
    print(f"  ⚠️  Remaining text columns in Set B: {remaining_object_B}")

print()
print("=" * 80)
print("🏁  Feature engineering complete!")
print("=" * 80)
