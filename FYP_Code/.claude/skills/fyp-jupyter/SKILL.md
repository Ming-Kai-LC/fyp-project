---
name: fyp-jupyter
description: Complete data science research workflow for Jupyter notebooks covering CRISP-DM methodology from data loading through model validation, with MLflow experiment tracking integration, phase-based workflow guidance (Exploration, Systematic Experimentation, Analysis, Documentation), and skill integration points. Use when working on FYP data science projects requiring systematic data preprocessing, EDA, feature engineering, modeling, statistical validation, experiment tracking, or needing guidance on what to work on at each project phase. Includes MLflow setup for tracking 30+ experiment runs, weekly work planning for 10-week FYP timeline, and clear decision framework for when to use which skill (fyp-jupyter, crossvit-covid19-fyp, fyp-statistical-validator, tar-umt-fyp-rds, tar-umt-academic-writing).
---

---

# FYP Jupyter Research Workflow

Complete guide for executing data science research in Jupyter notebooks following CRISP-DM methodology, optimized for Final Year Projects.

## Research Workflow Overview

This skill guides you through the complete data science pipeline:

**Phase 1: Setup & Data Loading** → **Phase 2: Data Cleaning** → **Phase 3: EDA** → **Phase 4: Feature Engineering** → **Phase 5: Modeling** → **Phase 6: Validation**

Each phase includes:
- **What to do**: Action checklist
- **Decision points**: When to use which method
- **Code templates**: Ready-to-run snippets
- **Validation**: How to verify correctness

For detailed methodology references, decision frameworks, and troubleshooting, see the `references/` folder.

## Phase 1: Setup & Data Loading

### Checklist
- [ ] Import required libraries (pandas, numpy, matplotlib, seaborn, sklearn)
- [ ] Set random seeds for reproducibility
- [ ] Load dataset using appropriate method
- [ ] Perform initial inspection

### Quick Template

```python
# Essential imports and setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Set random seeds for reproducibility
np.random.seed(42)

# Configure display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# Load data
df = pd.read_csv('data.csv')  # or read_excel, read_json, etc.

# Initial inspection
print(f"Dataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nBasic statistics:\n{df.describe()}")
print(f"\nDuplicates: {df.duplicated().sum()}")
```

### Decision Points

**Which file reading method?**
- CSV: `pd.read_csv()`
- Excel: `pd.read_excel()`
- JSON: `pd.read_json()`
- Large files (>500MB): Use `chunksize` parameter or `dask`

**Backup your raw data immediately:**
```python
df_raw = df.copy()  # Keep original untouched
```

## Phase 2: Data Cleaning & Preprocessing

### Checklist
- [ ] Handle duplicates
- [ ] Fix data types
- [ ] Handle missing values
- [ ] Detect and treat outliers
- [ ] Encode categorical variables
- [ ] Scale numerical features
- [ ] Document all transformations

### Missing Values Decision Framework

**Step 1: Assess the extent**
```python
missing_pct = (df.isnull().sum() / len(df)) * 100
print(missing_pct[missing_pct > 0].sort_values(ascending=False))
```

**Step 2: Choose strategy based on percentage:**

- **<5% missing + MCAR** → Listwise deletion (drop rows)
- **5-20% missing + numeric** → Median imputation (if skewed) or KNN
- **5-40% missing + mixed types** → MissForest (best accuracy)
- **Time series** → Forward/backward fill
- **>60% missing** → Drop column (unless critical)

**Quick implementations:**
```python
# Listwise deletion
df_clean = df.dropna()

# Median imputation
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

# KNN imputation
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
```

**For detailed methods including MissForest, see `references/preprocessing_methods.md`**

### Outlier Detection Decision Framework

**Recommended: Isolation Forest (no assumptions, works on high-dimensional data)**
```python
from sklearn.ensemble import IsolationForest

iso_forest = IsolationForest(contamination=0.05, random_state=42)
outliers = iso_forest.fit_predict(df[numeric_cols])
df_clean = df[outliers == 1]  # Keep inliers only
```

**Alternative methods:**
- **IQR method**: For univariate, quick checks
- **Z-score**: Only for normally distributed data
- **DBSCAN**: For spatial/cluster data

### Categorical Encoding Decision Framework

**Choose based on cardinality and model type:**

- **Binary (2 categories)**: Simple 0/1 mapping
- **Ordinal (ordered)**: Label encoding (0, 1, 2, ...)
- **Nominal + low cardinality (<10)**: One-hot encoding
- **Medium-high cardinality (10-50+)**: Target encoding (best performance)
- **Very high cardinality (>1000)**: Hash encoding

**Quick implementations:**
```python
# One-hot encoding (low cardinality)
df_encoded = pd.get_dummies(df, columns=['category_col'], drop_first=True)

# Target encoding (medium-high cardinality)
from sklearn.preprocessing import TargetEncoder
encoder = TargetEncoder(smooth='auto')
df['encoded_col'] = encoder.fit_transform(df[['category_col']], df['target'])
```

### Feature Scaling Decision Framework

**When to scale:**
- ✅ Linear/Logistic Regression, SVM, PCA, Neural Networks
- ❌ Tree-based models (Random Forest, XGBoost, Decision Trees)

**Which method:**
- **Default choice**: Standardization (z-score normalization)
- **Bounded range needed**: Min-max normalization
- **Outliers present**: RobustScaler

**Template:**
```python
from sklearn.preprocessing import StandardScaler

# CRITICAL: Fit on training data only to prevent leakage
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use same parameters
```

**For detailed preprocessing methods, see `references/preprocessing_methods.md`**

## Phase 3: Exploratory Data Analysis (EDA)

### Checklist
- [ ] Univariate analysis (distributions, statistics)
- [ ] Bivariate analysis (feature-target relationships)
- [ ] Multivariate analysis (correlations, interactions)
- [ ] Visualize key patterns
- [ ] Identify insights for feature engineering

### EDA Workflow

**Step 1: Univariate Analysis**
```python
# For numerical features
df[numeric_cols].hist(bins=30, figsize=(15, 10))
plt.tight_layout()
plt.show()

# For categorical features
for col in categorical_cols:
    print(f"\n{col} value counts:")
    print(df[col].value_counts())
    df[col].value_counts().plot(kind='bar')
    plt.show()
```

**Step 2: Bivariate Analysis**
```python
# Correlation with target
correlation = df[numeric_cols].corr()['target'].sort_values(ascending=False)
print(correlation)

# Visualize correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.show()

# Box plots for categorical vs numerical
for cat_col in categorical_cols:
    plt.figure(figsize=(10, 6))
    df.boxplot(column='target', by=cat_col)
    plt.title(f'Target distribution by {cat_col}')
    plt.show()
```

**Step 3: Multivariate Analysis**
```python
# Pairplot for key features
sns.pairplot(df[important_features + ['target']], hue='target')
plt.show()

# Scatter matrix
from pandas.plotting import scatter_matrix
scatter_matrix(df[numeric_cols], figsize=(15, 15), diagonal='kde')
plt.show()
```

### Visualization Selection Guide

**By data type combinations:**
- **Numerical → Numerical**: Scatter plot, line plot
- **Categorical → Numerical**: Box plot, violin plot, bar plot
- **Categorical → Categorical**: Stacked bar, heatmap
- **Distribution**: Histogram, KDE, box plot

**For comprehensive visualization guide and examples, see `references/eda_guide.md`**

## Phase 4: Feature Engineering

### Checklist
- [ ] Create interaction features
- [ ] Polynomial features (if needed)
- [ ] Domain-specific features
- [ ] Aggregate features
- [ ] Binning/discretization (if needed)

### Feature Engineering Templates

**Interaction features:**
```python
# Multiply related features
df['feature_interaction'] = df['feature1'] * df['feature2']

# Ratios
df['feature_ratio'] = df['numerator'] / (df['denominator'] + 1e-10)  # Avoid division by zero

# Differences
df['feature_diff'] = df['feature1'] - df['feature2']
```

**Polynomial features (use sparingly, increases dimensionality):**
```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X[['feature1', 'feature2']])
```

**Binning/discretization:**
```python
# Equal-width binning
df['age_group'] = pd.cut(df['age'], bins=5, labels=['Very Young', 'Young', 'Middle', 'Senior', 'Elderly'])

# Custom bins
df['income_category'] = pd.cut(df['income'], 
                               bins=[0, 30000, 60000, 100000, float('inf')],
                               labels=['Low', 'Medium', 'High', 'Very High'])
```

**Domain-specific features (example for time series):**
```python
df['datetime'] = pd.to_datetime(df['datetime'])
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['day_of_week'] = df['datetime'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
```

## Phase 5: Modeling

### Checklist
- [ ] Split data (train/validation/test)
- [ ] Select appropriate algorithm
- [ ] Train baseline model
- [ ] Tune hyperparameters
- [ ] Train final model
- [ ] Document random seeds and parameters

### Data Splitting Template

```python
from sklearn.model_selection import train_test_split

# Split: 70% train, 15% validation, 15% test
X = df.drop('target', axis=1)
y = df['target']

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y  # stratify for classification
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp  # 0.176 of 85% ≈ 15%
)

print(f"Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
```

### Model Selection Guide

**Classification tasks:**
- **Baseline**: Logistic Regression (fast, interpretable)
- **High performance**: Random Forest, XGBoost
- **Complex patterns**: Neural Networks (for large datasets)

**Regression tasks:**
- **Baseline**: Linear Regression (fast, interpretable)
- **High performance**: Random Forest, XGBoost
- **Complex patterns**: Neural Networks (for large datasets)

### Training Template

```python
# Example: XGBoost for classification
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Train model
model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_train, y_train)

# Validate
y_pred_val = model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_pred_val)
print(f"Validation Accuracy: {val_accuracy:.4f}")
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.3]
}

grid_search = GridSearchCV(
    XGBClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

# Use best model
best_model = grid_search.best_estimator_
```

## Phase 6: Validation & Results

### Checklist - CRITICAL FOR FYP
- [ ] Test final model on test set
- [ ] Calculate 95% confidence intervals
- [ ] Perform hypothesis testing
- [ ] Compare with baseline
- [ ] Generate confusion matrix / error metrics
- [ ] Calculate effect sizes
- [ ] Document all results properly

### Final Model Evaluation Template

```python
# Predict on test set
y_pred_test = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_test))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
```

### Computing 95% Confidence Intervals (REQUIRED FOR FYP)

```python
from scipy import stats

# For accuracy/performance metrics
def calculate_ci(scores, confidence=0.95):
    """Calculate confidence interval for a set of scores."""
    n = len(scores)
    mean = np.mean(scores)
    std_error = stats.sem(scores)
    margin = std_error * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, mean - margin, mean + margin

# Cross-validation for CI
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(best_model, X_train, y_train, cv=10, scoring='accuracy')
mean_acc, lower_ci, upper_ci = calculate_ci(cv_scores)

print(f"Mean Accuracy: {mean_acc:.4f}")
print(f"95% CI: [{lower_ci:.4f}, {upper_ci:.4f}]")
```

### Hypothesis Testing Template

```python
from scipy.stats import ttest_ind

# Compare your model vs baseline
baseline_scores = cross_val_score(baseline_model, X_train, y_train, cv=10)
model_scores = cross_val_score(best_model, X_train, y_train, cv=10)

# Paired t-test
t_stat, p_value = ttest_ind(model_scores, baseline_scores)

print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("✓ Statistically significant improvement (p < 0.05)")
else:
    print("✗ No significant improvement")
```

### Effect Size Calculation

```python
# Cohen's d
def cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    mean_diff = np.mean(group1) - np.mean(group2)
    pooled_std = np.sqrt((np.std(group1, ddof=1)**2 + np.std(group2, ddof=1)**2) / 2)
    return mean_diff / pooled_std

effect_size = cohens_d(model_scores, baseline_scores)
print(f"Cohen's d: {effect_size:.4f}")

# Interpretation
if abs(effect_size) < 0.2:
    print("Small effect")
elif abs(effect_size) < 0.5:
    print("Medium effect")
else:
    print("Large effect")
```

### Results Reporting Template

```python
# Comprehensive results summary
results = {
    'Model': 'XGBoost',
    'Test Accuracy': f"{test_accuracy:.4f}",
    '95% CI': f"[{lower_ci:.4f}, {upper_ci:.4f}]",
    'P-value': f"{p_value:.4f}",
    "Cohen's d": f"{effect_size:.4f}",
    'Statistical Significance': 'Yes' if p_value < 0.05 else 'No'
}

results_df = pd.DataFrame([results])
print(results_df)

# Save results
results_df.to_csv('model_results.csv', index=False)
```

**For comprehensive statistical validation methods and FYP requirements, see `references/model_validation.md`**
## MLflow Experiment Tracking

### Why MLflow for Your FYP?

**The Problem with Manual Logging:**
- Scattered results across multiple notebooks
- Forgetting which hyperparameters produced which results
- Difficult to compare 5 baseline models systematically
- Lost experiment configurations
- Manual table creation for thesis

**MLflow Solution:**
- Central experiment tracking (all runs in one place)
- Automatic parameter logging
- Easy comparison across runs
- Model versioning
- Direct integration with statistical-validator
- **Free and industry-standard** (used by Netflix, Databricks, etc.)

**Time Investment:** 5 minutes setup → Save 5-10 hours during FYP

### Quick Setup (5 Minutes)

```python
# Install MLflow
!pip install mlflow --break-system-packages

# Import and start tracking
import mlflow
import mlflow.sklearn  # or mlflow.pytorch, mlflow.tensorflow

# Create experiment (do this ONCE per project)
mlflow.set_experiment("crossvit-covid19-classification")

# That's it! Now you can log runs
```

### Basic Logging Pattern

```python
# Start a run (do this for EACH training session)
with mlflow.start_run(run_name="crossvit-seed-42"):
    
    # Log parameters (hyperparameters, config)
    mlflow.log_param("model", "CrossViT-Tiny")
    mlflow.log_param("learning_rate", 5e-5)
    mlflow.log_param("batch_size", 16)
    mlflow.log_param("epochs", 50)
    mlflow.log_param("random_seed", 42)
    mlflow.log_param("optimizer", "AdamW")
    
    # Train your model
    model.fit(X_train, y_train)
    
    # Log metrics (results)
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_metric("val_accuracy", val_acc)
    
    # Log the model itself
    mlflow.sklearn.log_model(model, "model")
    
    # Log artifacts (plots, confusion matrices)
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

print(f"Run logged! Accuracy: {accuracy:.4f}")
```

### Integration with Statistical Validator

**Step 1: Log multiple seeds in MLflow**
```python
seeds = [42, 123, 456, 789, 101112]
accuracies = []

for seed in seeds:
    with mlflow.start_run(run_name=f"crossvit-seed-{seed}"):
        # Log seed
        mlflow.log_param("random_seed", seed)
        mlflow.log_param("model", "CrossViT")
        
        # Train with this seed
        model = train_model(seed=seed)
        accuracy = evaluate(model, X_test, y_test)
        
        # Log result
        mlflow.log_metric("test_accuracy", accuracy)
        accuracies.append(accuracy)

print(f"All 5 seeds logged in MLflow!")
```

**Step 2: Calculate CI using statistical-validator**
```python
from scripts.confidence_intervals import multi_seed_ci

# Get accuracies from MLflow or use stored list
mean, lower, upper, std = multi_seed_ci(accuracies)

# Log aggregated results back to MLflow
with mlflow.start_run(run_name="crossvit-aggregated"):
    mlflow.log_param("aggregation", "5-seed-average")
    mlflow.log_metric("mean_accuracy", mean)
    mlflow.log_metric("ci_lower", lower)
    mlflow.log_metric("ci_upper", upper)
    mlflow.log_metric("std_dev", std)
```

**Step 3: View results in MLflow UI**
```bash
# In terminal, run:
mlflow ui

# Then open: http://localhost:5000
# You'll see all your experiments with sortable columns!
```

### MLflow Best Practices for FYP

**1. Naming Conventions**

Use consistent naming patterns:
```python
# Format: {model}-{variant}-seed-{number}
run_name = "crossvit-tiny-seed-42"
run_name = "resnet50-baseline-seed-42"
run_name = "densenet121-baseline-seed-123"
```

**2. What to Log**

**ALWAYS log:**
- All hyperparameters (lr, batch_size, epochs, etc.)
- Random seed
- Model architecture name
- Test accuracy
- Training/validation curves

**OPTIONALLY log:**
- Confusion matrix image
- ROC curve image
- Model checkpoint file
- Training logs

**DON'T log:**
- Raw datasets (too large, already have them)
- Every epoch (just final and best)
- Debug prints

**3. Organizing Experiments**

```python
# Different experiments for different model types
mlflow.set_experiment("crossvit-experiments")      # All CrossViT runs
mlflow.set_experiment("baseline-comparisons")      # All baseline runs
mlflow.set_experiment("ablation-study")           # Ablation experiments

# Or organize by FYP phase
mlflow.set_experiment("phase2-systematic-experiments")
mlflow.set_experiment("phase3-analysis-refinement")
```

**4. Comparing Models**

```python
# After logging all runs, compare them programmatically:
import mlflow

# Get all runs from experiment
experiment = mlflow.get_experiment_by_name("crossvit-experiments")
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

# View as DataFrame
print(runs[['params.model', 'metrics.test_accuracy', 'params.random_seed']].sort_values('metrics.test_accuracy', ascending=False))

# Export for thesis table
runs.to_csv('all_experiment_results.csv')
```

### Common Issues & Solutions

**Issue 1: "Connection refused" when running mlflow ui**
```bash
# Solution: Start MLflow server with specific port
mlflow ui --port 5001
```

**Issue 2: Runs not appearing in UI**
```python
# Solution: Check tracking URI
print(mlflow.get_tracking_uri())
# Should show: file:///path/to/mlruns

# Explicitly set if needed:
mlflow.set_tracking_uri("file:./mlruns")
```

**Issue 3: Too many runs cluttering the UI**
```python
# Solution: Use tags to filter
with mlflow.start_run():
    mlflow.set_tag("phase", "exploration")  # or "final"
    mlflow.set_tag("status", "complete")    # or "debugging"
    
# Filter in UI or programmatically:
runs = mlflow.search_runs(filter_string="tags.phase = 'final'")
```

**Issue 4: Forgot to log something**
```python
# Solution: Log to existing run
with mlflow.start_run(run_id="existing_run_id"):
    mlflow.log_metric("forgotten_metric", value)
```

### MLflow vs Manual Logging

| Aspect | Manual Logging | MLflow |
|--------|---------------|--------|
| Setup time | 0 min | 5 min |
| Log a run | 5-10 min | 30 sec |
| Compare runs | 30+ min | Instant (UI) |
| Find best model | Manual search | One click |
| Export for thesis | Manual tables | CSV export |
| **Total time saved** | - | **5-10 hours** |

### Integration with Chapter 5 Thesis Writing

```python
# After all experiments, export results for thesis
experiment = mlflow.get_experiment_by_name("final-experiments")
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

# Get data for statistical-validator
models = runs['params.model'].unique()
for model in models:
    model_runs = runs[runs['params.model'] == model]
    accuracies = model_runs['metrics.test_accuracy'].tolist()
    
    # Use statistical-validator to calculate CI
    from scripts.confidence_intervals import multi_seed_ci
    mean, lower, upper, std = multi_seed_ci(accuracies)
    
    print(f"{model}: {mean:.1%} (95% CI: {lower:.1%}-{upper:.1%})")
```

### Quick Reference

**Start tracking:**
```python
mlflow.set_experiment("experiment-name")
```

**Log a run:**
```python
with mlflow.start_run(run_name="descriptive-name"):
    mlflow.log_param("param_name", value)
    mlflow.log_metric("metric_name", value)
    mlflow.log_artifact("file.png")
```

**View results:**
```bash
mlflow ui
```

**Compare runs:**
```python
runs = mlflow.search_runs(experiment_ids=["id"])
print(runs[['params.model', 'metrics.test_accuracy']])
```

---
## Phase-Based FYP Workflow

### Overview: Your 10-Week Research Journey

FYP research follows a natural progression. This section helps you identify where you are and what to focus on each week.

**Timeline:**
- **Phase 1: Exploration** (Weeks 1-2) - Understanding the problem
- **Phase 2: Systematic Experimentation** (Weeks 3-6) - Running experiments
- **Phase 3: Analysis & Refinement** (Weeks 7-8) - Deep analysis
- **Phase 4: Documentation & Deployment** (Weeks 9-10) - Writing thesis

### Current Phase Identifier

**Answer these questions to identify your phase:**

1. Do you have a working baseline model? 
   - **No** → Phase 1 (Exploration)
   - **Yes** → Continue

2. Have you tested all 5 baseline models with statistical validation?
   - **No** → Phase 2 (Systematic Experimentation)
   - **Yes** → Continue

3. Have you completed hypothesis testing and CI calculations?
   - **No** → Phase 3 (Analysis & Refinement)
   - **Yes** → Phase 4 (Documentation)

### Phase 1: Exploration (Weeks 1-2)

**Goal:** Understand your dataset and get ONE baseline working

**What to Focus On:**
✅ Load and explore dataset
✅ Understand data distribution (class balance, image sizes, etc.)
✅ Implement basic preprocessing (CLAHE for your case)
✅ Get ONE simple baseline model working (e.g., ResNet-50)
✅ Verify training pipeline works end-to-end
✅ Document what you learned

**What NOT to Do:**
❌ Don't optimize hyperparameters yet
❌ Don't train with multiple seeds yet
❌ Don't compare models yet
❌ Don't worry about statistical validation yet

**Typical Notebooks:**
- `01_EDA.ipynb` - Dataset exploration
- `02_Preprocessing.ipynb` - Data pipeline
- `03_Baseline.ipynb` - First working model

**Success Criteria:**
- [ ] Can load dataset without errors
- [ ] Understand class distribution
- [ ] Have working train/val/test split
- [ ] ONE model trains to >70% accuracy
- [ ] Can generate predictions

**Daily Work Pattern:**
- **Day 1-2:** Dataset exploration, understand the data
- **Day 3-4:** Build preprocessing pipeline
- **Day 5-7:** Get first model training
- **Day 8-10:** Debug issues, verify reproducibility
- **Day 11-14:** Document findings, clean up code

**Key Question This Phase Answers:** 
"Can I successfully train a model on this dataset?"

**MLflow Usage:**
```python
# Just start logging, don't worry about organization yet
mlflow.set_experiment("exploration-phase")

with mlflow.start_run(run_name="first-baseline-resnet50"):
    mlflow.log_param("notes", "Initial attempt, getting pipeline working")
    # ... train and log
```

**Integration with Other Skills:**
- **crossvit-covid19-fyp** → Technical specifications
- **fyp-jupyter** (this skill) → Data exploration methods
- **tar-umt-fyp-rds** → Understanding FYP requirements

---

### Phase 2: Systematic Experimentation (Weeks 3-6)

**Goal:** Train ALL models (CrossViT + 5 baselines) with multiple seeds

**What to Focus On:**
✅ Train all 6 models (CrossViT + 5 baselines)
✅ Use 5 different random seeds for each model (30 total runs)
✅ Log everything in MLflow
✅ Save confusion matrices and predictions
✅ Document GPU memory usage and training time
✅ Keep organized experiment logs

**What NOT to Do:**
❌ Don't endless tune hyperparameters (use defaults)
❌ Don't add extra models beyond required 6
❌ Don't skip seeds (you NEED 5 for statistics)
❌ Don't start writing thesis yet

**Typical Notebooks:**
- `04_CrossViT_Training.ipynb` - Main model (5 seeds)
- `05_Baseline_ResNet.ipynb` - Baseline 1 (5 seeds)
- `06_Baseline_DenseNet.ipynb` - Baseline 2 (5 seeds)
- `07_Baseline_EfficientNet.ipynb` - Baseline 3 (5 seeds)
- `08_Baseline_ViT.ipynb` - Baseline 4 (5 seeds)
- `09_Baseline_Swin.ipynb` - Baseline 5 (5 seeds)

**Success Criteria:**
- [ ] All 6 models train successfully
- [ ] Each model has 5 seed runs
- [ ] All runs logged in MLflow
- [ ] Confusion matrices saved for each run
- [ ] Training metrics tracked
- [ ] Results reproducible

**Daily Work Pattern:**
- **Week 3:** Train CrossViT (5 seeds) - Your main model
- **Week 4:** Train baselines 1-2 (10 seeds total)
- **Week 5:** Train baselines 3-5 (15 seeds total)
- **Week 6:** Re-run any failed experiments, verify all results

**Key Question This Phase Answers:**
"How does CrossViT compare to baselines quantitatively?"

**MLflow Usage:**
```python
# Organized experiment tracking
mlflow.set_experiment("systematic-experiments")

seeds = [42, 123, 456, 789, 101112]
models = ["CrossViT", "ResNet50", "DenseNet121", "EfficientNetB0", "ViT-B32"]

for model_name in models:
    for seed in seeds:
        with mlflow.start_run(run_name=f"{model_name}-seed-{seed}"):
            mlflow.log_param("model", model_name)
            mlflow.log_param("random_seed", seed)
            mlflow.log_param("phase", "systematic-experimentation")
            
            # Train and log metrics
            accuracy = train_and_evaluate(model_name, seed)
            mlflow.log_metric("test_accuracy", accuracy)
```

**Integration with Other Skills:**
- **crossvit-covid19-fyp** → Model configurations
- **fyp-jupyter** (this skill, Phase 5) → Training templates
- MLflow section above → Experiment tracking

**Tips for This Phase:**
1. **Batch Processing:** Run overnight if possible
2. **Monitor GPU:** Use `nvidia-smi` to check memory
3. **Save Checkpoints:** Save best model from each run
4. **Document Issues:** Note any training problems
5. **Time Management:** Don't get stuck on one model

**If Things Go Wrong:**
- Model won't train? → Reduce batch size, check learning rate
- Out of memory? → Use mixed precision (FP16)
- Takes too long? → Run fewer epochs (30 instead of 50)
- Results poor? → Check preprocessing, verify data loading

---

### Phase 3: Analysis & Refinement (Weeks 7-8)

**Goal:** Statistical validation and deep analysis of results

**What to Focus On:**
✅ Calculate 95% CIs for all models
✅ Run hypothesis tests (paired t-test, McNemar's)
✅ Apply Bonferroni correction
✅ Generate comparison tables
✅ Analyze failure cases
✅ Document findings for thesis

**What NOT to Do:**
❌ Don't train new models (experiments done!)
❌ Don't change architectures
❌ Don't rerun everything (use saved results)
❌ Don't start thesis writing yet (analysis first!)

**Typical Notebooks:**
- `10_Statistical_Validation.ipynb` - CI and hypothesis testing
- `11_Results_Analysis.ipynb` - Deep dive into results
- `12_Error_Analysis.ipynb` - Understanding failures
- `13_Visualization.ipynb` - Generate all figures

**Success Criteria:**
- [ ] 95% CI calculated for all models
- [ ] Hypothesis tests completed (all comparisons)
- [ ] Bonferroni correction applied
- [ ] Results tables generated
- [ ] All figures created and saved
- [ ] Statistical significance confirmed

**Daily Work Pattern:**
- **Week 7 - Days 1-3:** Calculate CIs and run hypothesis tests
- **Week 7 - Days 4-7:** Generate tables and figures
- **Week 8 - Days 1-4:** Error analysis, understand why models fail/succeed
- **Week 8 - Days 5-7:** Prepare all outputs for thesis

**Key Question This Phase Answers:**
"Are the differences statistically significant, and why?"

**Statistical Validation Workflow:**
```python
# Step 1: Collect all results from MLflow
experiment = mlflow.get_experiment_by_name("systematic-experiments")
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

# Step 2: Organize by model
models = ["CrossViT", "ResNet50", "DenseNet121", "EfficientNetB0", "ViT-B32"]
results = {}

for model in models:
    model_runs = runs[runs['params.model'] == model]
    accuracies = model_runs['metrics.test_accuracy'].tolist()
    results[model] = accuracies

# Step 3: Calculate CIs for each model
from scripts.confidence_intervals import multi_seed_ci

ci_results = {}
for model, accuracies in results.items():
    mean, lower, upper, std = multi_seed_ci(accuracies)
    ci_results[model] = (mean, lower, upper)
    print(f"{model}: {mean:.1%} (95% CI: {lower:.1%}-{upper:.1%})")

# Step 4: Hypothesis testing (CrossViT vs each baseline)
from scripts.hypothesis_testing import paired_ttest, bonferroni_correction

crossvit_scores = results["CrossViT"]
p_values = []

for model in models[1:]:  # Skip CrossViT itself
    baseline_scores = results[model]
    t, p, sig, interp = paired_ttest(crossvit_scores, baseline_scores)
    p_values.append(p)
    print(f"CrossViT vs {model}: {interp}")

# Step 5: Multiple comparison correction
sig_flags, adj_alpha, interp = bonferroni_correction(p_values)
print(f"\n{interp}")

# Step 6: Generate thesis-ready table
from scripts.table_formatter import format_results_table

metrics_data = {
    "Accuracy": [ci_results[m] for m in models],
    # Add F1, AUC, etc.
}

table = format_results_table(models, metrics_data, "Model Performance Comparison", 1)
print(table)

# Save everything
with open('chapter5_table1.txt', 'w') as f:
    f.write(table)
```

**Integration with Other Skills:**
- **fyp-statistical-validator** → ALL validation functions
- **fyp-jupyter** (this skill) → Analysis templates
- MLflow → Retrieving experiment results

**Deliverables from This Phase:**
1. Statistical validation results (CIs, p-values)
2. Comparison tables (ready for thesis)
3. All figures (confusion matrices, ROC curves)
4. Error analysis findings
5. Organized results for Chapter 5

---

### Phase 4: Documentation & Deployment (Weeks 9-10)

**Goal:** Write thesis chapters and create Flask demo

**What to Focus On:**
✅ Write Chapter 4 (Research Design)
✅ Write Chapter 5 (Results and Evaluation)
✅ Create reproducibility statement
✅ Build simple Flask interface
✅ Prepare final presentation
✅ Final proofreading

**What NOT to Do:**
❌ Don't rerun experiments (too late!)
❌ Don't add new baselines
❌ Don't optimize Flask UI (basic is fine)
❌ Don't aim for perfection (aim for DONE)

**Typical Notebooks:**
- `14_Thesis_Content_Generation.ipynb` - Generate thesis content
- `15_Flask_Demo.ipynb` - Prototype interface
- `16_Final_Checks.ipynb` - Verify everything works

**Success Criteria:**
- [ ] Chapter 4 complete with reproducibility statement
- [ ] Chapter 5 complete with all tables and figures
- [ ] Flask demo works (basic but functional)
- [ ] All results reproducible
- [ ] Code organized and documented
- [ ] Ready to submit

**Daily Work Pattern:**
- **Week 9 - Days 1-3:** Chapter 4 (methodology, setup)
- **Week 9 - Days 4-7:** Chapter 5 (results, analysis)
- **Week 10 - Days 1-3:** Flask demo
- **Week 10 - Days 4-7:** Final checks, submission prep

**Key Question This Phase Answers:**
"Is everything ready to submit?"

**Chapter Generation Workflow:**
```python
# Generate reproducibility statement for Chapter 4
from scripts.reproducibility_generator import generate_reproducibility_statement

statement = generate_reproducibility_statement(
    random_seeds=[42, 123, 456, 789, 101112],
    n_runs=5,
    dataset_info={
        "name": "COVID-19 Radiography Database",
        "train_size": 16932,
        "val_size": 2116,
        "test_size": 2117,
        "n_classes": 4
    },
    model_config={
        "name": "CrossViT-Tiny",
        "input_size": "240x240",
        "parameters": "7M"
    },
    training_config={
        "epochs": 50,
        "batch_size": 16,
        "learning_rate": 5e-5,
        "optimizer": "AdamW"
    },
    hardware_info={
        "gpu": "NVIDIA RTX 4060",
        "vram": "8GB",
        "cpu": "AMD Ryzen 7",
        "ram": "32GB"
    }
)

# Save to file
with open('chapter4_section4.5.txt', 'w') as f:
    f.write(statement)

print("Chapter 4 reproducibility section saved!")
```

**Integration with Other Skills:**
- **fyp-statistical-validator** → Generate thesis content
- **fyp-chapter-bridge** (when created) → Automate chapter writing
- **tar-umt-academic-writing** → APA citations
- **tar-umt-fyp-rds** → Structure requirements

**Flask Demo (Keep It Simple):**
```python
# Minimal Flask app (30 lines total)
from flask import Flask, render_template, request, jsonify
import torch
from PIL import Image

app = Flask(__name__)
model = load_trained_model('best_crossvit.pth')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img = Image.open(file)
    prediction = model.predict(preprocess(img))
    return jsonify({'class': prediction, 'confidence': 0.92})

if __name__ == '__main__':
    app.run(debug=True)
```

**Final Submission Checklist:**
- [ ] All notebooks run top-to-bottom without errors
- [ ] Requirements.txt generated
- [ ] GitHub repository organized
- [ ] Chapter 4 complete
- [ ] Chapter 5 complete
- [ ] Flask demo works
- [ ] All figures saved
- [ ] All tables formatted
- [ ] Turnitin check passed (<20%)

---

### Phase Transition Guide

**Moving from Phase 1 → Phase 2:**
✅ You have: One working baseline
✅ You understand: Your dataset thoroughly
✅ Next action: Start systematic experiments with all models

**Moving from Phase 2 → Phase 3:**
✅ You have: All models trained (30 total runs)
✅ You understand: Which model performs best numerically
✅ Next action: Statistical validation

**Moving from Phase 3 → Phase 4:**
✅ You have: Validated results with CIs and p-values
✅ You understand: Why models differ statistically
✅ Next action: Write thesis chapters

**Stuck? Use This Decision Tree:**
```
Can't get model to train?
  → Phase 1: Debug preprocessing, reduce complexity
  
Models train but need all baselines?
  → Phase 2: Systematic experimentation
  
Have all results but need validation?
  → Phase 3: Statistical analysis
  
Results validated but need documentation?
  → Phase 4: Write thesis
```

---
## Skill Integration & When to Use What

### The Complete Skill Ecosystem

You now have access to multiple specialized skills. This section clarifies when to use each one and how they work together.

**Available Skills:**
1. **fyp-jupyter** (this skill) - Research workflow and experimentation
2. **crossvit-covid19-fyp** - Technical specifications and constraints
3. **fyp-statistical-validator** - Statistical validation and thesis formatting
4. **tar-umt-fyp-rds** - FYP structure and academic requirements
5. **tar-umt-academic-writing** - APA citations and plagiarism prevention

### When to Use: fyp-jupyter (This Skill)

**Use when you need to:**
- Understand what phase you're in (Exploration, Experimentation, etc.)
- Get daily work guidance ("What should I work on today?")
- Learn data preprocessing methods
- Set up MLflow experiment tracking
- Follow research workflow best practices
- Organize your notebooks systematically

**Examples:**
- "What phase am I in?" → fyp-jupyter
- "How do I log experiments in MLflow?" → fyp-jupyter (MLflow section)
- "What should I focus on this week?" → fyp-jupyter (Phase-based workflow)
- "How do I handle missing values?" → fyp-jupyter (Phase 2)
- "What's the CRISP-DM workflow?" → fyp-jupyter

**Don't use for:**
- Specific CrossViT technical details → Use crossvit-covid19-fyp
- Statistical validation calculations → Use fyp-statistical-validator
- FYP deadlines and deliverables → Use tar-umt-fyp-rds
- Writing thesis text → Use fyp-chapter-bridge (when created)

---

### When to Use: crossvit-covid19-fyp

**Use when you need to:**
- CrossViT model architecture details
- GPU memory constraints (RTX 4060 8GB)
- Dataset specifications (COVID-19 Radiography Database)
- CLAHE preprocessing parameters
- Training hyperparameters (learning rate, batch size, etc.)
- Baseline model configurations
- Hardware limitations

**Examples:**
- "What learning rate should I use for CrossViT?" → crossvit-covid19-fyp
- "What's the input size for CrossViT?" → crossvit-covid19-fyp (240×240)
- "How much VRAM does this need?" → crossvit-covid19-fyp (8GB constraint)
- "What are the baseline models?" → crossvit-covid19-fyp (5 specific models)
- "What CLAHE parameters?" → crossvit-covid19-fyp (clip_limit=2.0)

**Don't use for:**
- Workflow guidance → Use fyp-jupyter
- Statistical calculations → Use fyp-statistical-validator
- FYP structure questions → Use tar-umt-fyp-rds

---

### When to Use: fyp-statistical-validator

**Use when you need to:**
- Calculate 95% confidence intervals
- Perform hypothesis testing (McNemar's, paired t-test)
- Apply Bonferroni correction for multiple comparisons
- Generate APA-formatted results tables
- Create figure captions for confusion matrices
- Generate reproducibility statements for Chapter 4
- Format results for thesis chapters

**Examples:**
- "Calculate CI for my 5 seed runs" → fyp-statistical-validator
- "Is CrossViT significantly better than ResNet?" → fyp-statistical-validator (paired t-test)
- "Generate results table for Chapter 5" → fyp-statistical-validator
- "Create reproducibility statement" → fyp-statistical-validator
- "Format confusion matrix caption" → fyp-statistical-validator

**Don't use for:**
- Running experiments → Use fyp-jupyter
- Understanding FYP structure → Use tar-umt-fyp-rds
- Getting technical specs → Use crossvit-covid19-fyp

---

### When to Use: tar-umt-fyp-rds

**Use when you need to:**
- Understand FYP timeline and deadlines
- Know what deliverables are required
- Understand chapter structure (7 chapters)
- Learn about S.D.C.L. method for literature review
- Understand grading criteria
- Know supervisor meeting requirements
- Understand Forms 1, 2, 3 requirements

**Examples:**
- "When is Project I portfolio due?" → tar-umt-fyp-rds
- "What chapters are needed for Project I?" → tar-umt-fyp-rds (Chapters 1-4)
- "What's the S.D.C.L. method?" → tar-umt-fyp-rds
- "Do I need hypothesis in Chapter 1?" → tar-umt-fyp-rds (YES, mandatory)
- "How many feasibility studies?" → tar-umt-fyp-rds (All 5 required)

**Don't use for:**
- Running experiments → Use fyp-jupyter
- Statistical validation → Use fyp-statistical-validator
- Technical implementation → Use crossvit-covid19-fyp

---

### When to Use: tar-umt-academic-writing

**Use when you need to:**
- Format APA 7th Edition citations
- Check Turnitin similarity interpretation
- Learn proper paraphrasing techniques
- Understand plagiarism policies
- Format reference lists
- In-text citation formats

**Examples:**
- "How do I cite the COVID-19 dataset?" → tar-umt-academic-writing
- "Is 18% Turnitin similarity okay?" → tar-umt-academic-writing (Yes, <20%)
- "How to cite Chen et al. 2021?" → tar-umt-academic-writing
- "Paraphrasing vs quoting?" → tar-umt-academic-writing

**Don't use for:**
- Generating thesis content → Use fyp-statistical-validator or fyp-chapter-bridge
- Understanding FYP structure → Use tar-umt-fyp-rds

---

### Skill Usage Flowchart

```
START: I have a question
       ↓
   ┌───────────────────────────────────────────┐
   │ What type of question is it?              │
   └───────────────────────────────────────────┘
       ↓
       ├─→ "What should I work on today?"
       │   └─→ [fyp-jupyter] Phase-based workflow
       │
       ├─→ "How do I preprocess images?"
       │   └─→ [fyp-jupyter] Phase 2 preprocessing
       │
       ├─→ "What learning rate for CrossViT?"
       │   └─→ [crossvit-covid19-fyp] Model specs
       │
       ├─→ "How much VRAM do I need?"
       │   └─→ [crossvit-covid19-fyp] Hardware constraints
       │
       ├─→ "Calculate confidence intervals?"
       │   └─→ [fyp-statistical-validator] CI functions
       │
       ├─→ "Is my result statistically significant?"
       │   └─→ [fyp-statistical-validator] Hypothesis testing
       │
       ├─→ "When is my FYP due?"
       │   └─→ [tar-umt-fyp-rds] Deadlines
       │
       ├─→ "What chapters do I need?"
       │   └─→ [tar-umt-fyp-rds] Structure
       │
       └─→ "How to cite this paper?"
           └─→ [tar-umt-academic-writing] APA format
```

### Complete Workflow Integration

Here's how all skills work together in a typical FYP workflow:

**Week 1-2 (Phase 1: Exploration)**
```
1. [tar-umt-fyp-rds] → Understand FYP requirements
2. [crossvit-covid19-fyp] → Get technical specifications
3. [fyp-jupyter] → Set up notebooks and explore data
4. [fyp-jupyter] → Get first baseline working
```

**Week 3-6 (Phase 2: Systematic Experimentation)**
```
1. [fyp-jupyter] → Phase 2 workflow guidance
2. [crossvit-covid19-fyp] → Model configurations for all 6 models
3. [fyp-jupyter] → MLflow setup and logging
4. [fyp-jupyter] → Training templates and best practices
5. Run all experiments (30 total runs)
```

**Week 7-8 (Phase 3: Analysis & Refinement)**
```
1. [fyp-jupyter] → Phase 3 workflow guidance
2. [fyp-statistical-validator] → Calculate all CIs
3. [fyp-statistical-validator] → Run hypothesis tests
4. [fyp-statistical-validator] → Generate comparison tables
5. [fyp-jupyter] → Error analysis templates
```

**Week 9-10 (Phase 4: Documentation & Deployment)**
```
1. [tar-umt-fyp-rds] → Understand chapter requirements
2. [fyp-statistical-validator] → Generate reproducibility statement
3. [fyp-statistical-validator] → Format all thesis tables
4. [tar-umt-academic-writing] → Add APA citations
5. [fyp-jupyter] → Phase 4 workflow for Flask demo
```

### Example Scenarios

**Scenario 1: "I want to start my experiments"**
```
Step 1: [tar-umt-fyp-rds] → Understand what's required
Step 2: [fyp-jupyter] → Check which phase you're in
Step 3: [crossvit-covid19-fyp] → Get model specifications
Step 4: [fyp-jupyter] → Follow Phase 1 or 2 workflow
Step 5: [fyp-jupyter] → Set up MLflow logging
```

**Scenario 2: "I have results, need to validate"**
```
Step 1: [fyp-jupyter] → Confirm you're in Phase 3
Step 2: [fyp-statistical-validator] → Calculate CIs
Step 3: [fyp-statistical-validator] → Run hypothesis tests
Step 4: [fyp-statistical-validator] → Generate tables
```

**Scenario 3: "I need to write Chapter 5"**
```
Step 1: [tar-umt-fyp-rds] → Understand Chapter 5 structure
Step 2: [fyp-statistical-validator] → Generate all tables
Step 3: [fyp-statistical-validator] → Format figure captions
Step 4: [tar-umt-academic-writing] → Add citations
Step 5: [fyp-chapter-bridge] (future) → Auto-generate text
```

**Scenario 4: "My model won't train"**
```
Step 1: [crossvit-covid19-fyp] → Check hardware constraints
Step 2: [fyp-jupyter] → Review Phase 1 debugging tips
Step 3: [crossvit-covid19-fyp] → Verify batch size within limits
Step 4: [fyp-jupyter] → Try troubleshooting section
```

**Scenario 5: "I need to compare 5 baselines"**
```
Step 1: [crossvit-covid19-fyp] → Get baseline model specs
Step 2: [fyp-jupyter] → Use Phase 2 systematic workflow
Step 3: [fyp-jupyter] → Set up MLflow for all models
Step 4: [fyp-statistical-validator] → Calculate CIs for all
Step 5: [fyp-statistical-validator] → Compare with paired t-test
```

### Quick Decision Matrix

| Your Question | Primary Skill | Secondary Skills |
|--------------|---------------|------------------|
| What to work on today? | fyp-jupyter | tar-umt-fyp-rds |
| Model architecture details? | crossvit-covid19-fyp | - |
| Preprocessing methods? | fyp-jupyter | crossvit-covid19-fyp |
| Statistical validation? | fyp-statistical-validator | - |
| Thesis structure? | tar-umt-fyp-rds | - |
| Generate tables? | fyp-statistical-validator | - |
| Experiment tracking? | fyp-jupyter (MLflow) | - |
| APA citations? | tar-umt-academic-writing | - |
| Deadlines? | tar-umt-fyp-rds | - |
| Hardware limits? | crossvit-covid19-fyp | fyp-jupyter |
| Hypothesis testing? | fyp-statistical-validator | tar-umt-fyp-rds |

### Common Confusion: What's the Difference?

**fyp-jupyter vs crossvit-covid19-fyp:**
- **fyp-jupyter**: HOW to do research (workflow, methods, organization)
- **crossvit-covid19-fyp**: WHAT to implement (specs, configs, constraints)

**fyp-statistical-validator vs fyp-jupyter:**
- **fyp-statistical-validator**: Statistical calculations and thesis formatting
- **fyp-jupyter**: Research workflow and experiment management

**tar-umt-fyp-rds vs fyp-jupyter:**
- **tar-umt-fyp-rds**: FYP administrative requirements (structure, deadlines, forms)
- **fyp-jupyter**: Technical research execution (data science work)

### Integration Best Practices

**DO:**
✅ Use phase identifier from fyp-jupyter to guide daily work
✅ Reference crossvit-covid19-fyp for all technical decisions
✅ Use fyp-statistical-validator for ALL thesis results
✅ Check tar-umt-fyp-rds for deadlines weekly
✅ Use MLflow (from fyp-jupyter) for all experiments

**DON'T:**
❌ Ask fyp-jupyter for CrossViT specifications
❌ Ask crossvit-covid19-fyp for workflow guidance
❌ Calculate statistics manually (use fyp-statistical-validator)
❌ Guess chapter structure (check tar-umt-fyp-rds)
❌ Skip MLflow logging (you'll regret it later)

### Still Confused? Ask:

**"I need to [action]"**
- "work on experiments" → fyp-jupyter
- "understand CrossViT" → crossvit-covid19-fyp
- "validate results" → fyp-statistical-validator
- "know deadlines" → tar-umt-fyp-rds
- "cite sources" → tar-umt-academic-writing

**"I'm stuck on [problem]"**
- "what to do next" → fyp-jupyter (phase workflow)
- "model won't fit in memory" → crossvit-covid19-fyp (constraints)
- "don't know if significant" → fyp-statistical-validator
- "confused about FYP structure" → tar-umt-fyp-rds

---

### Summary: The Three Core Skills for Daily Work

**For most of your FYP work, you'll primarily use these 3:**

1. **fyp-jupyter** (this skill)
   - Your daily guide
   - What to work on
   - How to organize work

2. **crossvit-covid19-fyp**
   - Technical reference
   - Model specifications
   - Implementation details

3. **fyp-statistical-validator**
   - Results validation
   - Thesis formatting
   - Statistical rigor

**The other 2 skills support when needed:**
- **tar-umt-fyp-rds** → Administrative requirements
- **tar-umt-academic-writing** → Citation formatting

**Remember:** When in doubt, start with fyp-jupyter's phase identifier to know where you are, then use the appropriate skill for your specific need.

---
## Reproducibility Best Practices

### Critical Requirements
- Set random seeds at notebook start: `np.random.seed(42)`, `random_state=42`
- Document all package versions: `pip freeze > requirements.txt`
- Save trained models: `import joblib; joblib.dump(model, 'model.pkl')`
- Version control with Git (exclude large files in .gitignore)

### Notebook Organization

```python
# Cell 1: Imports and configuration
import pandas as pd
import numpy as np
np.random.seed(42)

# Cell 2: Load data
df = pd.read_csv('data.csv')

# Cell 3: Data cleaning
# ... (document each step with markdown)

# Cell 4: EDA
# ... (with clear explanations)

# And so on...
```

**Best practice**: Add markdown cells before each code cell explaining what you're doing and why.

### Testing Reproducibility

```python
# Before submission, restart kernel and run all cells
# Jupyter: Kernel → Restart & Run All
# Verify all cells execute without errors
# Verify results are identical to previous runs
```

## Common Pitfalls & Troubleshooting

### Data Leakage
❌ **Wrong**: Fit scaler on entire dataset
```python
scaler.fit(X)  # BAD: Includes test data
```

✅ **Correct**: Fit only on training data
```python
scaler.fit(X_train)  # GOOD: Only training data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Memory Issues
If working with large datasets:
```python
# Read in chunks
chunks = pd.read_csv('large_file.csv', chunksize=10000)
df = pd.concat([chunk for chunk in chunks])

# Or use dask for out-of-core computation
import dask.dataframe as dd
df = dd.read_csv('large_file.csv')
```

### Imbalanced Classes
```python
# Check class distribution
print(y.value_counts())

# Use stratified sampling
train_test_split(X, y, stratify=y, random_state=42)

# Or use class weights in model
model = XGBClassifier(scale_pos_weight=ratio, random_state=42)
```

**For more troubleshooting tips, see `references/troubleshooting.md`**

## Quick Reference Links

- **Detailed preprocessing methods**: See `references/preprocessing_methods.md`
- **Comprehensive EDA guide**: See `references/eda_guide.md`
- **Statistical validation & FYP requirements**: See `references/model_validation.md`
- **Ready-to-use code templates**: See `references/code_templates.md`
- **Common errors and fixes**: See `references/troubleshooting.md`

## FYP Submission Checklist

Before final submission:
- [ ] All notebooks run from top to bottom without errors
- [ ] Random seeds documented (np.random.seed, random_state parameters)
- [ ] Results include 95% confidence intervals
- [ ] Statistical significance tested (p-values)
- [ ] Effect sizes calculated (Cohen's d)
- [ ] Baseline comparison included
- [ ] All figures properly labeled and captioned
- [ ] Requirements.txt generated
- [ ] Raw data backed up and untouched
- [ ] All transformations documented
- [ ] Git repository properly maintained

This workflow ensures your FYP meets academic rigor requirements while developing practical industry-relevant skills.
