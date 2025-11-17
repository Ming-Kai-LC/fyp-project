# Data Preprocessing Methods - Comprehensive Reference

This reference provides detailed guidance on all preprocessing techniques for data cleaning and preparation.

## Missing Value Handling

### Understanding Missing Data Mechanisms

**MCAR (Missing Completely At Random)**
- Missing data is independent of any observed or unobserved data
- Example: Survey responses randomly lost due to technical errors
- Safe to delete rows with minimal bias

**MAR (Missing At Random)**
- Missing data depends on observed variables but not the missing value itself
- Example: Older patients less likely to report weight (age is observed)
- Can be handled with advanced imputation methods

**MNAR (Missing Not At Random)**
- Missing data relates to the unobserved value itself
- Example: High earners refusing to disclose income
- Most difficult to handle; requires domain knowledge

### Deletion Methods

**Listwise Deletion (Complete Case Analysis)**
```python
df_complete = df.dropna()
```
- **Use when**: <5% missing data under MCAR, large dataset (>10,000 samples)
- **Pros**: Simple, unbiased under MCAR
- **Cons**: Reduces sample size, biased if not MCAR

**Column Deletion**
```python
missing_threshold = 0.6
cols_to_drop = df.columns[df.isnull().mean() > missing_threshold]
df_cleaned = df.drop(columns=cols_to_drop)
```
- **Use when**: >60% missing in non-critical features
- **Pros**: Removes problematic variables
- **Cons**: Loss of potentially useful information

### Simple Imputation Methods

**Mean Imputation**
```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
```
- **Use when**: Normal distribution, <5% missing, continuous variables
- **Pros**: Fast, preserves sample size
- **Cons**: Reduces variance, distorts distribution, ignores relationships

**Median Imputation**
```python
imputer = SimpleImputer(strategy='median')
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
```
- **Use when**: Skewed distributions, outliers present, <5% missing
- **Pros**: Robust to outliers, fast
- **Cons**: Reduces variance, distorts distribution

**Mode Imputation**
```python
imputer = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = imputer.fit_transform(df[categorical_cols])
```
- **Use when**: Categorical variables, low cardinality
- **Pros**: Simple, preserves categorical nature
- **Cons**: Overrepresents most frequent category

**Forward/Backward Fill**
```python
# Time series only
df['value'] = df['value'].fillna(method='ffill')  # Forward fill
df['value'] = df['value'].fillna(method='bfill')  # Backward fill
```
- **Use when**: Time series data with temporal dependencies
- **Pros**: Maintains temporal patterns
- **Cons**: Only for ordered data, propagates errors

### Advanced Imputation Methods

**KNN Imputation**
```python
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5, weights='distance')
df_imputed = imputer.fit_transform(df[numeric_cols])
df[numeric_cols] = df_imputed
```
- **Use when**: 5-20% missing, MAR mechanism, want feature relationships preserved
- **Parameters**: 
  - `n_neighbors=5-10` (default 5)
  - `weights='distance'` for weighted average by proximity
- **Pros**: Considers feature relationships, works well for continuous variables
- **Cons**: Computationally expensive, sensitive to outliers, requires scaling

**MICE (Multiple Imputation by Chained Equations)**
```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = IterativeImputer(max_iter=10, random_state=42)
df_imputed = imputer.fit_transform(df[numeric_cols])
df[numeric_cols] = df_imputed
```
- **Use when**: 10-40% missing, complex relationships, MAR mechanism
- **Parameters**:
  - `max_iter=10` (iterations for convergence)
  - `estimator`: Can specify any sklearn estimator (default: BayesianRidge)
- **Pros**: Accounts for uncertainty, handles complex relationships, high accuracy
- **Cons**: Computationally intensive, requires substantial data

**MissForest (Random Forest Imputation)**
```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

# For numeric data
rf_imputer = IterativeImputer(
    estimator=RandomForestRegressor(n_estimators=10, random_state=42),
    max_iter=10,
    random_state=42
)
df_imputed = rf_imputer.fit_transform(df[numeric_cols])
df[numeric_cols] = df_imputed
```
- **Use when**: 5-40% missing, mixed data types, best accuracy needed
- **Pros**: Consistently lowest imputation error, handles non-linear relationships, works with mixed types
- **Cons**: Most computationally expensive, requires tuning

### Decision Framework for Missing Values

```
1. Check percentage missing:
   └─ <5% + MCAR → Listwise deletion
   └─ 5-20% + numeric → Median (if skewed) or KNN
   └─ 5-40% + mixed types → MissForest (best accuracy)
   └─ Time series → Forward/backward fill
   └─ >60% → Drop column (unless critical)

2. Check mechanism:
   └─ MCAR → Simple methods safe
   └─ MAR → Advanced methods (KNN, MICE, MissForest)
   └─ MNAR → Requires domain expertise

3. Balance accuracy vs. speed:
   └─ Need speed → KNN
   └─ Need accuracy → MissForest
   └─ Complex relationships → MICE
```

## Outlier Detection and Treatment

### Detection Methods

**Isolation Forest (Recommended for most cases)**
```python
from sklearn.ensemble import IsolationForest

iso_forest = IsolationForest(
    contamination=0.05,  # Expected proportion of outliers
    random_state=42,
    n_estimators=100
)
outliers = iso_forest.fit_predict(df[numeric_cols])

# -1 for outliers, 1 for inliers
df['is_outlier'] = outliers
df_clean = df[df['is_outlier'] == 1].drop('is_outlier', axis=1)
```
- **Use when**: High-dimensional data, no distributional assumptions needed
- **Parameters**: `contamination` (expected outlier proportion, typically 0.01-0.10)
- **Pros**: No assumptions, handles multivariate outliers, efficient
- **Cons**: Requires setting contamination parameter

**IQR Method**
```python
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (df[column] < lower_bound) | (df[column] > upper_bound)

# Apply to column
outlier_mask = detect_outliers_iqr(df, 'feature')
df_clean = df[~outlier_mask]
```
- **Use when**: Univariate analysis, quick checks, skewed data
- **Pros**: Simple, no distributional assumptions, visual interpretation
- **Cons**: Univariate only, arbitrary 1.5 multiplier

**Z-Score Method**
```python
from scipy import stats

z_scores = np.abs(stats.zscore(df[numeric_cols]))
outlier_mask = (z_scores > 3).any(axis=1)
df_clean = df[~outlier_mask]
```
- **Use when**: Data is approximately normally distributed
- **Pros**: Intuitive interpretation, standard approach
- **Cons**: Only for normal distributions, sensitive to extreme outliers in calculation

**DBSCAN (Density-Based Spatial Clustering)**
```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(df[numeric_cols])

# -1 indicates outliers
df_clean = df[labels != -1]
```
- **Use when**: Spatial data, clusters of varying density
- **Pros**: No contamination parameter needed, finds spatial outliers
- **Cons**: Requires parameter tuning (eps, min_samples)

### Treatment Strategies

**Removal**
```python
# Remove confirmed outliers
df_clean = df[~outlier_mask]
```
- **Use when**: Confirmed data errors, measurement mistakes
- **Caution**: Only remove if certain they're errors

**Capping (Winsorization)**
```python
# Cap at 1st and 99th percentiles
lower = df['feature'].quantile(0.01)
upper = df['feature'].quantile(0.99)
df['feature_capped'] = df['feature'].clip(lower=lower, upper=upper)
```
- **Use when**: Want to limit influence without removing data
- **Pros**: Retains sample size, reduces impact

**Transformation**
```python
# Log transformation
df['feature_log'] = np.log1p(df['feature'])

# Square root
df['feature_sqrt'] = np.sqrt(df['feature'])

# Box-Cox (requires positive values)
from scipy.stats import boxcox
df['feature_boxcox'], lambda_param = boxcox(df['feature'] + 1)
```
- **Use when**: Skewed distribution, need normalization
- **Pros**: Can make data more normal, improves model performance

**Separate Modeling**
- **Use when**: Outliers represent meaningful subpopulations
- **Approach**: Train separate models for normal and outlier groups

## Categorical Encoding

### Binary Encoding
```python
# Simple 0/1 mapping
df['gender_encoded'] = df['gender'].map({'Male': 0, 'Female': 1})
```
- **Use when**: Exactly 2 categories
- **Pros**: Simple, efficient
- **Cons**: Arbitrary ordering if categories are truly nominal

### Label/Ordinal Encoding
```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
df['category_encoded'] = encoder.fit_transform(df['category'])

# For ordinal with custom order
ordinal_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
df['priority_encoded'] = df['priority'].map(ordinal_mapping)
```
- **Use when**: Ordinal variables with natural order, tree-based models
- **Pros**: Memory efficient, preserves order, works well with tree models
- **Cons**: Implies false ordering for nominal variables with non-tree models

### One-Hot Encoding
```python
# Using pandas
df_encoded = pd.get_dummies(df, columns=['category'], drop_first=True)

# Using sklearn (returns array)
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_array = encoder.fit_transform(df[['category']])
encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out())
```
- **Use when**: Nominal variables, low cardinality (<10-15 categories), linear models
- **Parameters**: `drop='first'` to avoid dummy variable trap
- **Pros**: No ordinal assumptions, works with all models
- **Cons**: High dimensionality with high cardinality, sparse matrices

### Target Encoding (Best for High Cardinality)
```python
from sklearn.preprocessing import TargetEncoder

# With cross-validation to prevent leakage
encoder = TargetEncoder(smooth='auto', target_type='continuous')
df['category_target_encoded'] = encoder.fit_transform(df[['category']], df['target'])

# Manual implementation with smoothing
def target_encode_smooth(df, category_col, target_col, alpha=10):
    """Target encoding with smoothing to handle rare categories."""
    global_mean = df[target_col].mean()
    agg = df.groupby(category_col)[target_col].agg(['mean', 'count'])
    
    # Smooth: weight between category mean and global mean
    smooth_mean = (agg['mean'] * agg['count'] + global_mean * alpha) / (agg['count'] + alpha)
    return df[category_col].map(smooth_mean)

df['encoded'] = target_encode_smooth(df, 'category', 'target', alpha=10)
```
- **Use when**: Medium to high cardinality (10-50+ categories), strong target relationship
- **Parameters**: `smooth='auto'` for automatic smoothing, prevents overfitting
- **Pros**: Best performance in research, handles high cardinality efficiently, captures target relationship
- **Cons**: Risk of leakage without cross-validation, requires target variable

### Frequency Encoding
```python
freq_encoding = df['category'].value_counts().to_dict()
df['category_freq'] = df['category'].map(freq_encoding)
```
- **Use when**: Frequency correlates with target
- **Pros**: Simple, reduces dimensionality
- **Cons**: Different categories may have same frequency

### Hash Encoding
```python
from category_encoders import HashingEncoder

encoder = HashingEncoder(n_components=10)
df_encoded = encoder.fit_transform(df[['high_cardinality_col']])
```
- **Use when**: Very high cardinality (>1000 categories), unseen categories in test set
- **Pros**: Fixed dimensionality, handles unseen categories
- **Cons**: Hash collisions, less interpretable

## Feature Scaling

### Standardization (Z-score Normalization)
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use same parameters

# Manual calculation
mean = X_train.mean()
std = X_train.std()
X_train_scaled = (X_train - mean) / std
X_test_scaled = (X_test - mean) / std
```
- **Formula**: z = (x - μ) / σ
- **Result**: Mean=0, Std=1, unbounded range (typically -3 to +3)
- **Use when**: 
  - Linear/Logistic Regression
  - SVM
  - PCA
  - Neural Networks
  - Most gradient-based algorithms
- **Pros**: Less sensitive to outliers than min-max, works well with normal distributions
- **Cons**: Sensitive to outliers (less than min-max but still affected)

### Min-Max Normalization
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Custom range
scaler = MinMaxScaler(feature_range=(-1, 1))
```
- **Formula**: x' = (x - x_min) / (x_max - x_min)
- **Result**: Bounded range [0, 1] or custom range
- **Use when**:
  - k-NN
  - Neural networks requiring bounded inputs
  - Unknown distribution
  - Need specific range (e.g., [0, 1] for probabilities)
- **Pros**: Bounded range, preserves zero entries
- **Cons**: Very sensitive to outliers, compresses majority of values if outliers present

### RobustScaler (For Outliers)
```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```
- **Formula**: x' = (x - median) / IQR
- **Use when**: Dataset contains outliers that shouldn't be removed
- **Pros**: Robust to outliers, uses median and IQR instead of mean and std
- **Cons**: Not as interpretable as standard scaling

### MaxAbsScaler (For Sparse Data)
```python
from sklearn.preprocessing import MaxAbsScaler

scaler = MaxAbsScaler()
X_train_scaled = scaler.fit_transform(X_train)
```
- **Formula**: x' = x / max(|x|)
- **Result**: Range [-1, 1], preserves zeros
- **Use when**: Sparse data, need to preserve sparsity structure
- **Pros**: Preserves zero entries, good for sparse matrices
- **Cons**: Sensitive to outliers

### When NOT to Scale

**Tree-based models DO NOT require scaling:**
- Decision Trees
- Random Forest
- Gradient Boosting (XGBoost, LightGBM, CatBoost)
- Extra Trees

These models make splits based on thresholds and are invariant to feature scaling.

### Critical Practices

```python
# ✅ CORRECT: Fit on training data only
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

pipeline.fit(X_train, y_train)  # Scaler learns from training data only
predictions = pipeline.predict(X_test)  # Same parameters applied to test

# ❌ WRONG: Fitting on entire dataset causes data leakage
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # BAD: Includes test data information
X_train, X_test = train_test_split(X_scaled)  # Leakage already occurred
```

**Key principle**: Parameters for scaling (mean, std, min, max) must be computed ONLY from training data, then applied to test data using those same parameters.

## Practical Implementation Checklist

For any preprocessing task:
1. **Backup raw data**: `df_raw = df.copy()`
2. **Explore data characteristics**: Distribution, missingness, outliers
3. **Choose methods based on data properties**: Not all methods work for all data
4. **Document decisions**: Why you chose each method
5. **Validate transformations**: Check distributions before/after
6. **Prevent leakage**: Fit preprocessing on training data only
7. **Use pipelines**: Automate and ensure correct sequencing
8. **Test reproducibility**: Verify consistent results with same seeds
