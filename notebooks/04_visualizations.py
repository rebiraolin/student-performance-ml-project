"""
=============================================================================
 Student Performance Dataset — Model Visualizations
=============================================================================
 This script:
   1. Reproduces the trained Random Forest models from Script 03
   2. Creates Predicted vs Actual scatter plots for both models (side by side)
   3. Creates Feature Importance bar charts for both models
   4. Saves all plots as PNG files in the 'visualizations/' folder
   5. Includes beginner-friendly explanations in comments and printed output
=============================================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# ─────────────────────────────────────────────────────────────────────────────
# DATA PREPARATION & MODEL TRAINING  (reproduced from scripts 02 & 03)
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv("data/student-mat.csv", sep=";")

y = df["G3"]
X = df.drop(columns=["G3"])

X_with_grades    = X.copy()
X_without_grades = X.drop(columns=["G1", "G2"])

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

# Train/test split  (same random_state as Script 03 for consistency)
X_A_train, X_A_test, y_A_train, y_A_test = train_test_split(X_A, y, test_size=0.20, random_state=42)
X_B_train, X_B_test, y_B_train, y_B_test = train_test_split(X_B, y, test_size=0.20, random_state=42)

# Train models
model_A = RandomForestRegressor(n_estimators=100, random_state=42)
model_A.fit(X_A_train, y_A_train)

model_B = RandomForestRegressor(n_estimators=100, random_state=42)
model_B.fit(X_B_train, y_B_train)

# Predictions
y_A_test_pred = model_A.predict(X_A_test)
y_B_test_pred = model_B.predict(X_B_test)

# R² scores (for plot labels)
r2_A = r2_score(y_A_test, y_A_test_pred)
r2_B = r2_score(y_B_test, y_B_test_pred)

print("✅ Models trained successfully!")
print(f"   Model A (with G1 & G2) Test R²    : {r2_A:.4f}")
print(f"   Model B (without G1 & G2) Test R²  : {r2_B:.4f}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# Create output folder
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs("visualizations", exist_ok=True)

# =============================================================================
# PLOT 1 · Predicted vs Actual (side by side)
# =============================================================================
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  WHAT DOES THIS PLOT SHOW?                                             │
# │                                                                        │
# │  Each dot represents ONE STUDENT in the test set.                      │
# │    • X-axis = Actual grade (what the student really scored)            │
# │    • Y-axis = Predicted grade (what the model thinks they scored)      │
# │                                                                        │
# │  The red dashed line is the "perfect prediction" line (y = x).         │
# │  If every prediction were exactly right, ALL dots would sit on this    │
# │  line.                                                                 │
# │                                                                        │
# │  HOW TO READ IT:                                                       │
# │    • Dots close to the line      → accurate predictions               │
# │    • Dots far from the line      → wrong predictions                  │
# │    • Dots above the line         → model OVER-estimates the grade     │
# │    • Dots below the line         → model UNDER-estimates the grade    │
# │    • Tight cluster around line   → good model                         │
# │    • Scattered dots everywhere   → weak model                         │
# └──────────────────────────────────────────────────────────────────────────┘

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- Model A (with G1 & G2) ---
ax1.scatter(y_A_test, y_A_test_pred, alpha=0.6, edgecolors="k", linewidths=0.5,
            color="#4C72B0", s=60, label="Students")
ax1.plot([0, 20], [0, 20], "r--", linewidth=2, label="Perfect prediction")
ax1.set_xlabel("Actual G3 (Final Grade)", fontsize=12)
ax1.set_ylabel("Predicted G3", fontsize=12)
ax1.set_title(f"Model A — With G1 & G2\nTest R² = {r2_A:.4f}", fontsize=13, fontweight="bold")
ax1.set_xlim(-0.5, 20.5)
ax1.set_ylim(-0.5, 20.5)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# --- Model B (without G1 & G2) ---
ax2.scatter(y_B_test, y_B_test_pred, alpha=0.6, edgecolors="k", linewidths=0.5,
            color="#DD8452", s=60, label="Students")
ax2.plot([0, 20], [0, 20], "r--", linewidth=2, label="Perfect prediction")
ax2.set_xlabel("Actual G3 (Final Grade)", fontsize=12)
ax2.set_ylabel("Predicted G3", fontsize=12)
ax2.set_title(f"Model B — Without G1 & G2\nTest R² = {r2_B:.4f}", fontsize=13, fontweight="bold")
ax2.set_xlim(-0.5, 20.5)
ax2.set_ylim(-0.5, 20.5)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

fig1.suptitle("Predicted vs Actual Final Grade (G3)", fontsize=15, fontweight="bold", y=1.02)
fig1.tight_layout()
fig1.savefig("visualizations/predicted_vs_actual.png", dpi=150, bbox_inches="tight")
print("📊 Saved: visualizations/predicted_vs_actual.png")

# =============================================================================
# PLOT 2 · Feature Importance — Model A (with G1 & G2)
# =============================================================================
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  WHAT IS FEATURE IMPORTANCE?                                           │
# │                                                                        │
# │  Random Forest can tell us HOW MUCH each feature contributed to its    │
# │  predictions. This is called "feature importance".                     │
# │                                                                        │
# │  • A feature with HIGH importance strongly influences the model's      │
# │    predictions — changing it would significantly change the output.    │
# │  • A feature with LOW importance has little effect — the model         │
# │    barely uses it.                                                     │
# │                                                                        │
# │  The importance values sum to 1.0 (100%). They are calculated by       │
# │  measuring how much each feature reduces prediction error across all   │
# │  100 trees in the forest.                                              │
# │                                                                        │
# │  WHY IS THIS USEFUL?                                                   │
# │    • Understand WHAT DRIVES student performance.                       │
# │    • Identify features you could REMOVE without losing accuracy.       │
# │    • Generate ACTIONABLE INSIGHTS (e.g. "study time matters more       │
# │      than internet access").                                           │
# └──────────────────────────────────────────────────────────────────────────┘

def plot_feature_importance(model, feature_names, title, filename, color):
    """Create a horizontal bar chart of the top 15 most important features."""

    # Get importance values from the trained model
    importances = model.feature_importances_

    # Create a DataFrame and sort by importance (descending)
    feat_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=True)

    # Show only the top 15 features (for readability)
    top_n = 15
    feat_imp_top = feat_imp.tail(top_n)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(feat_imp_top["Feature"], feat_imp_top["Importance"],
                   color=color, edgecolor="k", linewidth=0.5)

    # Add value labels on each bar
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{width:.3f}", va="center", fontsize=9)

    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"visualizations/{filename}", dpi=150, bbox_inches="tight")
    print(f"📊 Saved: visualizations/{filename}")

    # Return the sorted DataFrame for printing
    return feat_imp.sort_values("Importance", ascending=False)

# --- Feature importance: Model A ---
print()
imp_A = plot_feature_importance(
    model_A,
    X_A.columns.tolist(),
    "Feature Importance — Model A (with G1 & G2)",
    "feature_importance_model_A.png",
    color="#4C72B0"
)

# --- Feature importance: Model B ---
imp_B = plot_feature_importance(
    model_B,
    X_B.columns.tolist(),
    "Feature Importance — Model B (without G1 & G2)",
    "feature_importance_model_B.png",
    color="#DD8452"
)

# =============================================================================
# PRINTED ANALYSIS — Feature Importance Rankings
# =============================================================================
print()
print("=" * 80)
print("TOP 10 MOST IMPORTANT FEATURES")
print("=" * 80)

print()
print("  Model A (with G1 & G2):")
print("  " + "-" * 40)
for i, (_, row) in enumerate(imp_A.head(10).iterrows(), 1):
    bar = "█" * int(row["Importance"] * 100)
    print(f"    {i:2d}. {row['Feature']:20s}  {row['Importance']:.4f}  {bar}")

print()
print("  Model B (without G1 & G2):")
print("  " + "-" * 40)
for i, (_, row) in enumerate(imp_B.head(10).iterrows(), 1):
    bar = "█" * int(row["Importance"] * 100)
    print(f"    {i:2d}. {row['Feature']:20s}  {row['Importance']:.4f}  {bar}")

# =============================================================================
# BEGINNER-FRIENDLY INTERPRETATION
# =============================================================================
print()
print("=" * 80)
print("INTERPRETATION (BEGINNER-FRIENDLY)")
print("=" * 80)
print("""
  PREDICTED vs ACTUAL PLOTS
  ─────────────────────────
    • Model A: Dots cluster tightly around the red line → predictions are
      very close to reality. G1 and G2 give the model strong signals.

    • Model B: Dots are more scattered → predictions are rougher. Without
      prior grades, the model struggles to pinpoint the exact final score.

  FEATURE IMPORTANCE — MODEL A (with G1 & G2)
  ────────────────────────────────────────────
    • G2 (2nd period grade) is likely the #1 feature — it's the most recent
      grade before the final exam, so it's the best predictor.
    • G1 (1st period grade) is likely #2 — earlier but still very informative.
    • Together, G1 and G2 dominate, making other features far less important.

  FEATURE IMPORTANCE — MODEL B (without G1 & G2)
  ────────────────────────────────────────────
    • With grades removed, the model has to rely on all the other factors.
    • Features like 'failures' (past class failures), 'absences', 'higher'
      (desire for higher education), 'age', and 'studytime' typically rise
      to the top.
    • This view is more ACTIONABLE for educators — it reveals which
      non-academic factors most strongly influence student outcomes.

  KEY TAKEAWAY
  ────────────
    • G1 and G2 are so powerful that they overshadow everything else.
    • For UNDERSTANDING student behaviour, Model B's importance chart
      is more insightful (it tells you what factors matter BEYOND grades).
    • For PREDICTION ACCURACY, Model A is superior.
""")

print("=" * 80)
print("🏁  Visualization complete!  Check the 'visualizations/' folder for PNG files.")
print("=" * 80)
