# Heart Disease Prediction

A machine learning project to predict the presence of heart disease in patients based on clinical and diagnostic attributes.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [ML Pipeline](#ml-pipeline)
- [Model Results](#model-results)
- [Author](#author)

---

## Project Overview

This project builds an end-to-end ML classification pipeline to predict whether a patient has heart disease based on clinical measurements and diagnostic test results. The dataset contains 303 patient records with features spanning age, sex, chest pain type, blood pressure, cholesterol, heart rate, and other cardiac indicators.

The target variable is `target` — a binary label indicating the presence (1) or absence (0) of heart disease. The pipeline covers the full ML workflow: data loading, missing value handling, categorical encoding, outlier treatment, feature scaling, feature selection, model training, and accuracy evaluation across 7 classification models.

---

## Repository Structure

```
heart-disease-prediction/
├── heart_prediction.ipynb        # End-to-end ML pipeline notebook
└── heart.csv                     # Dataset (303 patient records)
```

---

## Dataset

**File:** `heart.csv`  
**Source:** Kaggle  
**Total Records:** 303  
**Target Variable:** `target` (0 = No Heart Disease, 1 = Heart Disease) — 220 vs 83

| Feature | Type | Description |
|---|---|---|
| `age` | Integer | Age of the patient (29 – 77) |
| `sex` | Integer | Sex of the patient (0 = Female, 1 = Male) |
| `cp` | Integer | Chest pain type (1 – 4) |
| `trestbps` | Integer | Resting blood pressure in mm Hg |
| `chol` | Integer | Serum cholesterol in mg/dl |
| `fbs` | Integer | Fasting blood sugar > 120 mg/dl (0 = No, 1 = Yes) |
| `restecg` | Integer | Resting ECG results (0 – 2) |
| `thalach` | Integer | Maximum heart rate achieved |
| `exang` | Integer | Exercise induced angina (0 = No, 1 = Yes) |
| `oldpeak` | Float | ST depression induced by exercise relative to rest |
| `slope` | Integer | Slope of the peak exercise ST segment (1 – 3) |
| `ca` | Integer | Number of major vessels coloured by fluoroscopy (0 – 3) |
| `thal` | Categorical | Thalassemia type (fixed / normal / reversible) |
| `target` | Integer | Target — presence of heart disease (0 / 1) |

### Summary Statistics

| Feature | Mean | Std Dev | Min | Max |
|---|---|---|---|---|
| Age | 54.59 | 9.02 | 29 | 77 |
| Resting BP | 131.69 | 17.60 | 94 | 200 |
| Cholesterol | 246.69 | 51.78 | 126 | 564 |
| Max Heart Rate | 149.61 | 22.88 | 71 | 202 |

---

## ML Pipeline

### 1. Data Loading and Exploration

- Loaded dataset using `pandas`
- Inspected shape, data types, and missing value counts
- Reviewed feature distributions and target class balance

### 2. Missing Value Handling

- **Numerical columns** — imputed using mean via `SimpleImputer(strategy='mean')`
- **Categorical columns** — imputed using mode via `SimpleImputer(strategy='most_frequent')`

### 3. Categorical Encoding

- Applied **Label Encoding** on binary columns using `LabelEncoder`
- Applied **One-Hot Encoding** on multi-class column (`thal` — fixed / normal / reversible) using `pd.get_dummies()` with `drop_first=True` to avoid multicollinearity

### 4. Outlier Treatment

- Used **IQR (Interquartile Range)** method for all numerical columns
- Capped values below `Q1 - 1.5 * IQR` to the lower bound
- Capped values above `Q3 + 1.5 * IQR` to the upper bound

### 5. Feature Selection

- Applied `SelectKBest` with `f_classif` (ANOVA F-score) to identify the top 10 most relevant features
- Selected features: `age`, `sex`, `cp`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal_normal`, `thal_reversible`
- Visualized feature correlations using a heatmap (`seaborn`)

### 6. Feature Scaling

- Applied `StandardScaler` to normalize all feature values to zero mean and unit variance before model training

### 7. Train-Test Split

| Parameter | Value |
|---|---|
| Test size | 30% (91 records) |
| Train size | 70% (212 records) |
| Random state | 4 |

---

## Model Results

All 7 classification models were trained on the same preprocessed dataset and evaluated using **Accuracy Score**.

| # | Model | Accuracy |
|---|---|---|
| 1 | **Logistic Regression** | **86.81%** |
| 2 | Naive Bayes | 83.52% |
| 3 | Gradient Boosting | 80.22% |
| 4 | Support Vector Machine | 79.12% |
| 5 | Random Forest | 78.02% |
| 6 | K-Nearest Neighbors | 75.82% |
| 7 | Decision Tree | 74.73% |

**Best performing model: Logistic Regression — 86.81% accuracy**

---

## Author

**Bremikha Arunachalam**  
GitHub: [github.com/BremikhaArunachalam](https://github.com/BremikhaArunachalam)
