# Ready-to-Use Code Templates

This reference provides complete, ready-to-run code templates for common data science workflows in Jupyter notebooks.

## Complete Workflow Template

```python
# ===================================================================
# COMPLETE DATA SCIENCE WORKFLOW TEMPLATE
# ===================================================================

# 1. IMPORTS AND SETUP
# ===================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

# Configure display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
plt.style.use('seaborn-v0_8-darkgrid')

# 2. LOAD DATA
# ===================================================================
df = pd.read_csv('data.csv')
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Backup raw data
df_raw = df.copy()

# 3. INITIAL EXPLORATION
# ===================================================================
print("\n=== Dataset Info ===")
print(df.info())

print("\n=== First Few Rows ===")
print(df.head())

print("\n=== Statistical Summary ===")
print(df.describe())

print("\n=== Missing Values ===")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({'Missing Count': missing, 'Percentage': missing_pct})
print(missing_df[missing_df['Missing Count'] > 0])

print("\n=== Duplicate Rows ===")
print(f"Number of duplicates: {df.duplicated().sum()}")

# 4. DATA CLEANING
# ===================================================================

# Remove duplicates
df = df.drop_duplicates()

# Handle missing values (choose appropriate method)
# Option 1: Drop rows with missing values (<5% missing)
# df = df.dropna()

# Option 2: Impute missing values
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Impute numeric columns with median
if numeric_cols:
    imputer_numeric = SimpleImputer(strategy='median')
    df[numeric_cols] = imputer_numeric.fit_transform(df[numeric_cols])

# Impute categorical columns with mode
if categorical_cols:
    imputer_categorical = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = imputer_categorical.fit_transform(df[categorical_cols])

print(f"\nCleaned dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# 5. EXPLORATORY DATA ANALYSIS (EDA)
# ===================================================================

# Numeric features distribution
if len(numeric_cols) > 0:
    df[numeric_cols].hist(bins=30, figsize=(15, 10), edgecolor='black')
    plt.suptitle('Distribution of Numeric Features', y=1.02, fontsize=16)
    plt.tight_layout()
    plt.show()

# Correlation matrix
if len(numeric_cols) > 2:
    plt.figure(figsize=(12, 10))
    corr_matrix = df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=1)
    plt.title('Feature Correlation Matrix', fontsize=16)
    plt.tight_layout()
    plt.show()

# Target variable analysis (if exists)
if 'target' in df.columns:
    print("\n=== Target Variable Distribution ===")
    print(df['target'].value_counts())
    
    plt.figure(figsize=(8, 6))
    df['target'].value_counts().plot(kind='bar', edgecolor='black')
    plt.title('Target Variable Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

# 6. FEATURE ENGINEERING
# ===================================================================

# Encode categorical variables
le = LabelEncoder()
for col in categorical_cols:
    if col != 'target':  # Don't encode target yet
        df[f'{col}_encoded'] = le.fit_transform(df[col])
        print(f"Encoded {col}: {df[col].unique()} -> {df[f'{col}_encoded'].unique()}")

# Drop original categorical columns (keep encoded versions)
df = df.drop(columns=categorical_cols, errors='ignore')

# 7. PREPARE DATA FOR MODELING
# ===================================================================

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Split data: 70% train, 15% validation, 15% test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
)

print(f"\nTraining set: {len(X_train)} samples")
print(f"Validation set: {len(X_val)} samples")
print(f"Test set: {len(X_test)} samples")

# Scale features (if needed for the model)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 8. MODEL TRAINING
# ===================================================================

from sklearn.ensemble import RandomForestClassifier

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)
print("\n✓ Model trained successfully")

# 9. MODEL EVALUATION
# ===================================================================

# Validate on validation set
y_val_pred = model.predict(X_val_scaled)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"\nValidation Accuracy: {val_accuracy:.4f}")

# Final test on test set
y_test_pred = model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Detailed classification report
print("\n=== Classification Report ===")
print(classification_report(y_test, y_test_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# Cross-validation
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=10, scoring='accuracy')
print(f"\n=== Cross-Validation Results ===")
print(f"CV Scores: {cv_scores}")
print(f"Mean CV Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

# 10. SAVE RESULTS
# ===================================================================

import joblib

# Save model
joblib.dump(model, 'trained_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\n✓ Model and scaler saved")

# Save results
results = {
    'Metric': ['Validation Accuracy', 'Test Accuracy', 'CV Mean Accuracy', 'CV Std Dev'],
    'Value': [f"{val_accuracy:.4f}", f"{test_accuracy:.4f}", 
              f"{np.mean(cv_scores):.4f}", f"{np.std(cv_scores):.4f}"]
}
results_df = pd.DataFrame(results)
results_df.to_csv('results.csv', index=False)
print("✓ Results saved to 'results.csv'")
```

## Binary Classification Template

```python
# Binary Classification Workflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Set random seed
np.random.seed(42)

# Load data
df = pd.read_csv('data.csv')

# Prepare data
X = df.drop('target', axis=1)
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

## Multiclass Classification Template

```python
# Multiclass Classification Workflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Evaluate
print("=== Classification Report ===")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Per-class accuracy
from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred, output_dict=True)
for label, metrics in report.items():
    if isinstance(metrics, dict):
        print(f"Class {label}: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
```

## Regression Template

```python
# Regression Workflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Set random seed
np.random.seed(42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Prediction vs Actual plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.grid(alpha=0.3)
plt.show()

# Residuals plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid(alpha=0.3)
plt.show()
```

## Hyperparameter Tuning Template

```python
# Grid Search for Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create model
rf = RandomForestClassifier(random_state=42)

# Grid search
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

# Fit grid search
print("Starting grid search...")
grid_search.fit(X_train_scaled, y_train)

# Best parameters
print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

# Use best model
best_model = grid_search.best_estimator_

# Evaluate on test set
y_pred = best_model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {test_accuracy:.4f}")
```

## Cross-Validation Template

```python
# Comprehensive Cross-Validation
from sklearn.model_selection import cross_val_score, cross_validate
from scipy import stats

# Simple cross-validation
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=10, scoring='accuracy')

print("=== Cross-Validation Results ===")
print(f"CV Scores: {cv_scores}")
print(f"Mean: {np.mean(cv_scores):.4f}")
print(f"Std Dev: {np.std(cv_scores):.4f}")

# Calculate 95% confidence interval
n = len(cv_scores)
mean = np.mean(cv_scores)
std_error = stats.sem(cv_scores)
margin = std_error * stats.t.ppf(0.975, n - 1)
ci_lower = mean - margin
ci_upper = mean + margin

print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

# Multiple metrics
scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
cv_results = cross_validate(model, X_train_scaled, y_train, cv=10, scoring=scoring)

print("\n=== Multiple Metrics ===")
for metric in scoring:
    scores = cv_results[f'test_{metric}']
    print(f"{metric}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
```

## Feature Importance Template

```python
# Feature Importance Analysis
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Get feature importance
importance = model.feature_importances_
feature_names = X_train.columns

# Create dataframe
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
}).sort_values('Importance', ascending=False)

print("=== Feature Importance ===")
print(importance_df)

# Plot top features
plt.figure(figsize=(10, 8))
plt.barh(importance_df['Feature'].head(15), importance_df['Importance'].head(15))
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top 15 Most Important Features')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Using SHAP for more detailed explanation (if installed)
try:
    import shap
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Summary plot
    shap.summary_plot(shap_values, X_test, plot_type="bar")
    
    # Detailed plot for first class (binary) or all classes (multiclass)
    if len(shap_values) == 2:
        shap.summary_plot(shap_values[1], X_test)
    else:
        shap.summary_plot(shap_values, X_test)
        
except ImportError:
    print("SHAP not installed. Install with: pip install shap")
```

## Handling Imbalanced Data Template

```python
# Handling Imbalanced Classes
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier

# Check class distribution
print("Class distribution:")
print(y_train.value_counts())
print(f"\nClass proportions:")
print(y_train.value_counts(normalize=True))

# Method 1: Class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

model_weighted = RandomForestClassifier(
    n_estimators=100,
    class_weight=class_weight_dict,
    random_state=42
)
model_weighted.fit(X_train_scaled, y_train)

# Method 2: SMOTE (Synthetic Minority Over-sampling)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

print(f"\nAfter SMOTE:")
print(pd.Series(y_train_smote).value_counts())

model_smote = RandomForestClassifier(n_estimators=100, random_state=42)
model_smote.fit(X_train_smote, y_train_smote)

# Method 3: Random Under-sampling
rus = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train_scaled, y_train)

print(f"\nAfter Random Under-sampling:")
print(pd.Series(y_train_rus).value_counts())

model_rus = RandomForestClassifier(n_estimators=100, random_state=42)
model_rus.fit(X_train_rus, y_train_rus)

# Compare methods
for name, model in [('Weighted', model_weighted), ('SMOTE', model_smote), ('Under-sampled', model_rus)]:
    y_pred = model.predict(X_test_scaled)
    print(f"\n{name} Model:")
    print(classification_report(y_test, y_pred))
```

## Model Comparison Template

```python
# Compare Multiple Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
}

# Train and evaluate each model
results = []

for name, model in models.items():
    print(f"Training {name}...")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    
    # Train on full training set
    model.fit(X_train_scaled, y_train)
    
    # Test set performance
    y_pred = model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    results.append({
        'Model': name,
        'CV Mean': f"{np.mean(cv_scores):.4f}",
        'CV Std': f"{np.std(cv_scores):.4f}",
        'Test Accuracy': f"{test_accuracy:.4f}"
    })

# Display results
results_df = pd.DataFrame(results)
print("\n=== Model Comparison ===")
print(results_df)

# Save results
results_df.to_csv('model_comparison.csv', index=False)
```

## Pipeline Template

```python
# Complete Pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# Create pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train pipeline
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# Cross-validation with pipeline
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
print(f"CV Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

# Save pipeline
import joblib
joblib.dump(pipeline, 'pipeline.pkl')

# Load and use pipeline
loaded_pipeline = joblib.load('pipeline.pkl')
predictions = loaded_pipeline.predict(X_test)
```

## Time Series Split Template

```python
# Time Series Cross-Validation
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

# Time series split
tscv = TimeSeriesSplit(n_splits=5)

# Visualize splits
fig, ax = plt.subplots(figsize=(12, 5))

for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
    ax.plot(train_idx, [i] * len(train_idx), 'b-', linewidth=10, label='Train' if i == 0 else '')
    ax.plot(test_idx, [i] * len(test_idx), 'r-', linewidth=10, label='Test' if i == 0 else '')

ax.set_xlabel('Sample Index')
ax.set_ylabel('CV Split')
ax.set_title('Time Series Cross-Validation Splits')
ax.legend()
plt.tight_layout()
plt.show()

# Cross-validation with time series split
cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
print(f"Time Series CV Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
```

## Save and Load Models Template

```python
# Save and Load Models
import joblib
import pickle

# Method 1: Joblib (recommended for sklearn models)
joblib.dump(model, 'model.pkl')
loaded_model = joblib.load('model.pkl')

# Method 2: Pickle (general purpose)
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Save multiple objects
joblib.dump({
    'model': model,
    'scaler': scaler,
    'feature_names': X_train.columns.tolist()
}, 'model_package.pkl')

# Load multiple objects
package = joblib.load('model_package.pkl')
model = package['model']
scaler = package['scaler']
feature_names = package['feature_names']

# Verify loaded model
y_pred = loaded_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Loaded model accuracy: {accuracy:.4f}")
```

## Learning Curve Template

```python
# Learning Curve Analysis
from sklearn.model_selection import learning_curve

# Generate learning curve
train_sizes, train_scores, val_scores = learning_curve(
    model, X_train_scaled, y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# Calculate mean and std
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Training score', color='blue', marker='o')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, val_mean, label='Validation score', color='red', marker='o')
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color='red')

plt.xlabel('Training Set Size')
plt.ylabel('Accuracy Score')
plt.title('Learning Curve')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Interpretation
if train_mean[-1] - val_mean[-1] > 0.1:
    print("⚠️ High variance (overfitting): training score much higher than validation")
    print("   → Try: reduce model complexity, add regularization, get more data")
elif val_mean[-1] < 0.7:
    print("⚠️ High bias (underfitting): both scores are low")
    print("   → Try: increase model complexity, add features, remove regularization")
else:
    print("✓ Model looks good: training and validation scores are close and high")
```

## Quick Start Minimal Template

```python
# Minimal Quick Start (Classification)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load data
df = pd.read_csv('data.csv')
X = df.drop('target', axis=1)
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```
