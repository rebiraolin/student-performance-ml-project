# Predicting Student Performance Using Machine Learning

## Project Overview
This project predicts the **final grade (G3)** of students using a dataset from student performance records.  
The goal is to understand which factors influence student outcomes and explore realistic prediction scenarios using machine learning.

---

## Dataset
- Source: [Student Performance Dataset (CSV)](https://archive.ics.uci.edu/ml/datasets/Student+Performance)  
- Size: 395 rows × 33 columns  
- Column types:
  - **Numeric:** age, studytime, absences, grades (G1, G2, G3), etc.  
  - **Categorical:** school, sex, address, family size, parent jobs, etc.  
- Target variable: `G3` (final grade, 0–20)

---

## Project Workflow

### 1. Data Exploration
- Loaded CSV using pandas  
- Previewed first 10 rows and basic dataset info  
- Checked for missing values (none present)  
- Explained each column in beginner-friendly terms  
- Identified `G3` as the target variable

**Notebook:** `notebooks/01_data_exploration.py`

---

### 2. Feature Engineering
- Created two feature sets:
  - **Feature Set A:** includes `G1` & `G2` (midterm grades)  
  - **Feature Set B:** excludes `G1` & `G2` (early prediction scenario)  
- Encoded categorical features:
  - **Binary features:** 0/1 mapping (e.g., sex: M→0, F→1)  
  - **Multi-category features:** one-hot encoding (`Mjob`, `Fjob`, `reason`, `guardian`)  
- Separated features (`X`) and target (`y`)

**Notebook:** `notebooks/02_feature_engineering.py`

---

### 3. Model Training
- Train/test split: 80/20 (random_state=42)  
- Model: Random Forest Regressor (100 trees)  
- Trained separately on both feature sets  
- Evaluated with **R² score**:
  - Model A (with G1 & G2) → high accuracy  
  - Model B (without G1 & G2) → realistic early prediction, lower accuracy  
- Insights:
  - Including prior grades makes prediction easier  
  - Feature Set B highlights important behavioral and demographic factors

**Notebook:** `notebooks/03_model_training.py`

---

### 4. Visualization & Feature Insights
- **Predicted vs Actual plots:** scatter plots show model accuracy  
- **Feature importance charts:** identify which features influence predictions most  
  - Model A: `G1` & `G2` dominate  
  - Model B: failures, absences, studytime, higher education aspiration  
- Beginner-friendly explanations included

**Notebook:** `notebooks/04_visualizations.py`  
**Visualizations:** `visualizations/` folder (PNG files)

---

## Key Insights
1. Prior grades (G1 & G2) are the strongest predictors of final grades.  
2. Without prior grades, factors like failures, absences, and study habits become more important.  
3. Realistic early predictions are harder but provide actionable insights for interventions.

---

## How to Run
1. Clone the repo:  
```bash
git clone https://github.com/rebiraolin/student-performance-ml-project.git
cd student-performance-ml-project
