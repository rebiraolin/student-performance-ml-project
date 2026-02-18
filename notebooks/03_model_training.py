"""
=============================================================================
 Student Performance Dataset — Model Training & Evaluation
=============================================================================
 This script:
   1. Reuses the feature engineering from Script 02 (encoding, two sets)
   2. Splits each feature set into 80% training / 20% testing
   3. Trains a Random Forest Regressor on both feature sets
   4. Evaluates each model with R² score (train & test)
   5. Compares the two models and explains the results
=============================================================================
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# ─────────────────────────────────────────────────────────────────────────────
# DATA PREPARATION  (reproduced from 02_feature_engineering.py for self-
#                     contained execution — same logic, compact form)
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv("data/student-mat.csv", sep=";")

# Separate target (y) and features (X)
y = df["G3"]
X = df.drop(columns=["G3"])

# Two feature sets
X_with_grades    = X.copy()                       # Set A: includes G1, G2
X_without_grades = X.drop(columns=["G1", "G2"])   # Set B: excludes G1, G2

# Encoding helpers
binary_mappings = {
    "school": {"GP": 0, "MS": 1},   "sex":        {"F": 0, "M": 1},
    "address": {"R": 0, "U": 1},    "famsize":    {"LE3": 0, "GT3": 1},
    "Pstatus": {"A": 0, "T": 1},    "schoolsup":  {"no": 0, "yes": 1},
    "famsup":  {"no": 0, "yes": 1}, "paid":       {"no": 0, "yes": 1},
    "activities": {"no": 0, "yes": 1}, "nursery": {"no": 0, "yes": 1},
    "higher":  {"no": 0, "yes": 1}, "internet":   {"no": 0, "yes": 1},
    "romantic": {"no": 0, "yes": 1},
}
multi_cols = ["Mjob", "Fjob", "reason", "guardian"]

def encode_features(X_input):
    X_enc = X_input.copy()
    for col, mapping in binary_mappings.items():
        if col in X_enc.columns:
            X_enc[col] = X_enc[col].map(mapping)
    X_enc = pd.get_dummies(X_enc, columns=multi_cols, drop_first=True)
    return X_enc

X_A = encode_features(X_with_grades)
X_B = encode_features(X_without_grades)

print("✅ Dataset loaded & encoded!")
print(f"   Feature Set A (with G1 & G2) : {X_A.shape}")
print(f"   Feature Set B (without G1 & G2): {X_B.shape}")
print(f"   Target (y)                    : {y.shape}")
print()

# =============================================================================
# STEP 1 · Train / Test Split
# =============================================================================
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  WHY DO WE SPLIT THE DATA?                                             │
# │                                                                        │
# │  Imagine studying for an exam using ONLY last year's exam paper, then  │
# │  taking the SAME paper as your real exam. You'd score 100% — but you   │
# │  haven't actually learned the material.                                │
# │                                                                        │
# │  The same thing happens with ML models. If we evaluate a model on the  │
# │  SAME data it learned from, it looks artificially perfect. That's      │
# │  called OVERFITTING — memorising the answers instead of learning       │
# │  the patterns.                                                         │
# │                                                                        │
# │  By holding out 20% of the data that the model NEVER sees during       │
# │  training, we get an honest measure of how well it would perform on    │
# │  new, unseen students.                                                 │
# │                                                                        │
# │  • Training set (80%) → the model learns patterns from this data.      │
# │  • Testing  set (20%) → we check the model's predictions here.         │
# │                                                                        │
# │  random_state=42 makes the split reproducible — you'll get the same    │
# │  split every time you run the script, making results comparable.       │
# └──────────────────────────────────────────────────────────────────────────┘

# Split Feature Set A (with G1 & G2)
X_A_train, X_A_test, y_A_train, y_A_test = train_test_split(
    X_A, y, test_size=0.20, random_state=42
)

# Split Feature Set B (without G1 & G2)  — SAME split seed for fair comparison
X_B_train, X_B_test, y_B_train, y_B_test = train_test_split(
    X_B, y, test_size=0.20, random_state=42
)

print("=" * 80)
print("STEP 1 — TRAIN / TEST SPLIT  (80% train · 20% test)")
print("=" * 80)
print(f"  Feature Set A:")
print(f"    Training : {X_A_train.shape[0]} samples × {X_A_train.shape[1]} features")
print(f"    Testing  : {X_A_test.shape[0]} samples × {X_A_test.shape[1]} features")
print()
print(f"  Feature Set B:")
print(f"    Training : {X_B_train.shape[0]} samples × {X_B_train.shape[1]} features")
print(f"    Testing  : {X_B_test.shape[0]} samples × {X_B_test.shape[1]} features")
print()

# =============================================================================
# STEP 2 · Train a Random Forest Regressor for each feature set
# =============================================================================
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  WHAT IS A RANDOM FOREST?                                              │
# │                                                                        │
# │  Think of a "decision tree" as a flowchart of yes/no questions:        │
# │    "Is study time > 5 hrs?"  →  "Are absences < 10?"  →  prediction   │
# │                                                                        │
# │  A single tree can overfit (memorise noise). A Random Forest builds    │
# │  MANY trees (default: 100), each trained on a slightly different       │
# │  random subset of the data. The final prediction is the AVERAGE of     │
# │  all trees — this "wisdom of the crowd" approach is much more robust.  │
# │                                                                        │
# │  Key parameters:                                                       │
# │    n_estimators = 100   → number of trees in the forest                │
# │    random_state = 42    → reproducible results                         │
# └──────────────────────────────────────────────────────────────────────────┘

print("=" * 80)
print("STEP 2 — TRAINING RANDOM FOREST REGRESSORS")
print("=" * 80)

# --- Model A: with G1 & G2 ---
print("  Training Model A (with G1 & G2) ... ", end="")
model_A = RandomForestRegressor(n_estimators=100, random_state=42)
model_A.fit(X_A_train, y_A_train)
print("done ✅")

# --- Model B: without G1 & G2 ---
print("  Training Model B (without G1 & G2) ... ", end="")
model_B = RandomForestRegressor(n_estimators=100, random_state=42)
model_B.fit(X_B_train, y_B_train)
print("done ✅")
print()

# =============================================================================
# STEP 3 · Evaluate with R² score
# =============================================================================
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  WHAT IS THE R² SCORE?                                                 │
# │                                                                        │
# │  R² (R-squared), also called the "coefficient of determination",       │
# │  tells you what PERCENTAGE of the variation in the target (G3) your    │
# │  model explains.                                                       │
# │                                                                        │
# │    R² = 1.00  →  Perfect: the model explains 100% of the variation.   │
# │    R² = 0.80  →  Good: the model explains 80% of the variation.       │
# │    R² = 0.00  →  Bad: the model is no better than always predicting   │
# │                   the average grade.                                   │
# │    R² < 0.00  →  Terrible: the model is WORSE than just guessing      │
# │                   the average.                                         │
# │                                                                        │
# │  We calculate R² on BOTH train and test sets:                          │
# │    • Train R² shows how well the model learned the training data.      │
# │    • Test  R² shows how well it generalises to new data.               │
# │    • If Train R² ≫ Test R², the model is OVERFITTING.                 │
# └──────────────────────────────────────────────────────────────────────────┘

# Predictions — Model A
y_A_train_pred = model_A.predict(X_A_train)
y_A_test_pred  = model_A.predict(X_A_test)

# Predictions — Model B
y_B_train_pred = model_B.predict(X_B_train)
y_B_test_pred  = model_B.predict(X_B_test)

# R² scores
r2_A_train = r2_score(y_A_train, y_A_train_pred)
r2_A_test  = r2_score(y_A_test,  y_A_test_pred)
r2_B_train = r2_score(y_B_train, y_B_train_pred)
r2_B_test  = r2_score(y_B_test,  y_B_test_pred)

print("=" * 80)
print("STEP 3 — R² SCORES")
print("=" * 80)
print()
print("  ┌────────────────────────────────────────────────────────┐")
print("  │  Model              │  Train R²     │  Test R²        │")
print("  ├────────────────────────────────────────────────────────┤")
print(f"  │  A (with G1 & G2)   │  {r2_A_train:.4f}        │  {r2_A_test:.4f}          │")
print(f"  │  B (without G1 & G2)│  {r2_B_train:.4f}        │  {r2_B_test:.4f}          │")
print("  └────────────────────────────────────────────────────────┘")
print()

# =============================================================================
# STEP 4 · Interpretation & comparison
# =============================================================================
print("=" * 80)
print("STEP 4 — INTERPRETATION")
print("=" * 80)
print(f"""
  MODEL A  (with G1 & G2)
  ─────────────────────────
    Train R² = {r2_A_train:.4f}   |   Test R²  = {r2_A_test:.4f}
    → G1 and G2 are HIGHLY predictive of the final grade (G3).
      This makes sense: a student's earlier exam scores are the
      strongest indicator of their final score.

  MODEL B  (without G1 & G2)
  ─────────────────────────
    Train R² = {r2_B_train:.4f}   |   Test R²  = {r2_B_test:.4f}
    → Without prior grades, the model relies only on demographics,
      family, study habits, and social factors. Accuracy drops
      significantly.

  KEY TAKEAWAY
  ────────────
    • Including G1 & G2 gives MUCH better predictions because past
      grades are the strongest predictor of future grades.
    • Without G1 & G2, performance drops — but this scenario is more
      useful for EARLY PREDICTION (before any exams are taken).
    • If Train R² is much higher than Test R², the model may be
      overfitting (memorising training data instead of learning
      general patterns).
""")

# Overfitting check
gap_A = r2_A_train - r2_A_test
gap_B = r2_B_train - r2_B_test

print("  OVERFITTING CHECK")
print("  ─────────────────")
print(f"    Model A: Train−Test gap = {gap_A:.4f}", end="")
print("  ← some overfitting" if gap_A > 0.10 else "  ← acceptable")
print(f"    Model B: Train−Test gap = {gap_B:.4f}", end="")
print("  ← some overfitting" if gap_B > 0.10 else "  ← acceptable")
print()

print("=" * 80)
print("🏁  Model training & evaluation complete!")
print("=" * 80)
