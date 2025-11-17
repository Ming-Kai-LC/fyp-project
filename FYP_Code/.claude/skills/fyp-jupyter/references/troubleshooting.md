# Troubleshooting Guide - Common Errors and Fixes

This reference provides solutions to common errors encountered in data science workflows.

## Import Errors

### ModuleNotFoundError: No module named 'X'

**Error:**
```
ModuleNotFoundError: No module named 'sklearn'
```

**Solution:**
```bash
# Install missing package
pip install scikit-learn

# Or for conda
conda install scikit-learn

# For multiple packages
pip install scikit-learn pandas numpy matplotlib seaborn

# Install from requirements.txt
pip install -r requirements.txt
```

### ImportError: cannot import name 'X' from 'Y'

**Error:**
```
ImportError: cannot import name 'train_test_split' from 'sklearn'
```

**Solution:**
```python
# Wrong
from sklearn import train_test_split

# Correct
from sklearn.model_selection import train_test_split
```

## Data Loading Errors

### FileNotFoundError

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'data.csv'
```

**Solution:**
```python
import os

# Check current directory
print("Current directory:", os.getcwd())

# List files in current directory
print("Files:", os.listdir())

# Use absolute path
df = pd.read_csv('/full/path/to/data.csv')

# Or use relative path correctly
df = pd.read_csv('./data/data.csv')
```

### UnicodeDecodeError when reading CSV

**Error:**
```
UnicodeDecodeError: 'utf-8' codec can't decode byte
```

**Solution:**
```python
# Try different encodings
df = pd.read_csv('data.csv', encoding='latin1')
# or
df = pd.read_csv('data.csv', encoding='cp1252')
# or
df = pd.read_csv('data.csv', encoding='iso-8859-1')

# Auto-detect encoding
import chardet

with open('data.csv', 'rb') as f:
    result = chardet.detect(f.read())
    encoding = result['encoding']

df = pd.read_csv('data.csv', encoding=encoding)
```

### ParserError: Error tokenizing data

**Error:**
```
ParserError: Error tokenizing data. C error: Expected X fields, saw Y
```

**Solution:**
```python
# Option 1: Specify error_bad_lines parameter
df = pd.read_csv('data.csv', on_bad_lines='skip')

# Option 2: Check delimiter
df = pd.read_csv('data.csv', sep=';')  # or '\t' for tab

# Option 3: Read with more flexibility
df = pd.read_csv('data.csv', engine='python', on_bad_lines='skip')
```

## Data Type Errors

### ValueError: could not convert string to float

**Error:**
```
ValueError: could not convert string to float: 'NA'
```

**Solution:**
```python
# Replace non-numeric values before conversion
df['column'] = pd.to_numeric(df['column'], errors='coerce')  # Converts invalid to NaN

# Or clean specific values
df['column'] = df['column'].replace(['NA', 'N/A', 'null'], np.nan)
df['column'] = df['column'].astype(float)

# Remove non-numeric characters
df['column'] = df['column'].str.replace('[^0-9.]', '', regex=True)
df['column'] = pd.to_numeric(df['column'])
```

### TypeError: unsupported operand type(s)

**Error:**
```
TypeError: unsupported operand type(s) for -: 'str' and 'float'
```

**Solution:**
```python
# Check data types
print(df.dtypes)
print(df['column'].dtype)

# Convert to correct type
df['column'] = df['column'].astype(float)
# or
df['column'] = pd.to_numeric(df['column'], errors='coerce')
```

## Missing Value Errors

### ValueError: Input contains NaN

**Error:**
```
ValueError: Input contains NaN, infinity or a value too large
```

**Solution:**
```python
# Check for missing values
print(df.isnull().sum())

# Option 1: Remove missing values
df = df.dropna()

# Option 2: Impute missing values
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')  # or 'median', 'most_frequent'
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

# Option 3: Fill with specific value
df = df.fillna(0)  # or df.fillna(df.mean())

# Check for infinity
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()
```

## Shape Mismatch Errors

### ValueError: X has Y features but Z is expecting N

**Error:**
```
ValueError: X has 10 features but classifier is expecting 15
```

**Solution:**
```python
# Ensure same features in train and test
# Option 1: Check column names match
print("Train columns:", X_train.columns.tolist())
print("Test columns:", X_test.columns.tolist())

# Option 2: Reindex to match
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Option 3: Ensure transformations are applied consistently
# WRONG: Different transformations
scaler1.fit_transform(X_train)
scaler2.fit_transform(X_test)  # Creates different features!

# CORRECT: Same transformation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Uses same parameters
```

### ValueError: Found input variables with inconsistent numbers of samples

**Error:**
```
ValueError: Found input variables with inconsistent numbers of samples: [100, 90]
```

**Solution:**
```python
# Check shapes
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Ensure same length
assert len(X) == len(y), f"Length mismatch: X={len(X)}, y={len(y)}"

# Common causes:
# 1. Forgot to apply same filtering to X and y
# WRONG
X = df.drop('target', axis=1)
y = df['target']
X = X[X['age'] > 18]  # Filters X but not y!

# CORRECT
df_filtered = df[df['age'] > 18]
X = df_filtered.drop('target', axis=1)
y = df_filtered['target']

# 2. Index mismatch after operations
# Reset indices
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)
```

## Memory Errors

### MemoryError: Unable to allocate array

**Error:**
```
MemoryError: Unable to allocate 5.73 GiB for an array
```

**Solution:**
```python
# Option 1: Use chunking for large files
chunks = []
for chunk in pd.read_csv('large_file.csv', chunksize=10000):
    # Process each chunk
    processed = process(chunk)
    chunks.append(processed)
df = pd.concat(chunks)

# Option 2: Use Dask for out-of-core computation
import dask.dataframe as dd
df = dd.read_csv('large_file.csv')

# Option 3: Reduce data types
df['int_col'] = df['int_col'].astype('int32')  # Instead of int64
df['float_col'] = df['float_col'].astype('float32')  # Instead of float64

# Option 4: Sample data for development
df_sample = df.sample(frac=0.1, random_state=42)  # Use 10% of data

# Option 5: Free memory
import gc
del large_object
gc.collect()
```

## Model Training Errors

### ValueError: Classification metrics can't handle a mix of binary and continuous targets

**Error:**
```
ValueError: Classification metrics can't handle a mix of binary and continuous targets
```

**Solution:**
```python
# Check if y is continuous when it should be categorical
print(y.unique())  # Should show discrete classes
print(y.dtype)

# Convert to integer if needed
y = y.astype(int)

# Or use regression metrics instead
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
```

### ValueError: Number of classes in y_true not equal to the number of columns in probabilities

**Error:**
```
ValueError: Number of classes, 2, does not match size of target_names, 3
```

**Solution:**
```python
# Ensure target_names matches actual classes
from sklearn.metrics import classification_report

# Get actual classes
classes = np.unique(y_test)
print(f"Actual classes: {classes}")

# Use correct class names
target_names = [f'Class {i}' for i in classes]
print(classification_report(y_test, y_pred, target_names=target_names))

# Or let it auto-generate
print(classification_report(y_test, y_pred))
```

### RuntimeWarning: overflow encountered in exp

**Solution:**
```python
# This usually happens with extreme values
# Solution: Scale your features

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Or clip extreme values
X = np.clip(X, -10, 10)
```

## Data Leakage Issues

### Problem: Test accuracy much higher than expected

**Cause: Data leakage - information from test set leaking into training**

**Common mistakes and fixes:**

```python
# WRONG: Scaling before split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Leakage! Test data info used
X_train, X_test = train_test_split(X_scaled)

# CORRECT: Scale after split
X_train, X_test = train_test_split(X)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit on train only
X_test_scaled = scaler.transform(X_test)  # Transform test with train parameters

# WRONG: Imputing before split
imputer = SimpleImputer()
X_imputed = imputer.fit_transform(X)  # Leakage!
X_train, X_test = train_test_split(X_imputed)

# CORRECT: Impute after split
X_train, X_test = train_test_split(X)
imputer = SimpleImputer()
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# WRONG: Feature selection before split
selector = SelectKBest(k=10)
X_selected = selector.fit_transform(X, y)  # Leakage!
X_train, X_test = train_test_split(X_selected)

# CORRECT: Select features after split
X_train, X_test, y_train, y_test = train_test_split(X, y)
selector = SelectKBest(k=10)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)
```

## Index Errors

### KeyError: 'column_name'

**Error:**
```
KeyError: 'target'
```

**Solution:**
```python
# Check if column exists
print(df.columns.tolist())

# Case-sensitive check
print('target' in df.columns)
print('Target' in df.columns)

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Rename columns if needed
df = df.rename(columns={'old_name': 'new_name'})
```

### IndexError: single positional indexer is out-of-bounds

**Solution:**
```python
# Check dataframe length
print(f"DataFrame length: {len(df)}")

# Use .loc or .iloc safely
if len(df) > 0:
    first_row = df.iloc[0]
else:
    print("DataFrame is empty")

# Use .at for single value access (faster and safer)
value = df.at[0, 'column_name']
```

## Convergence Warnings

### ConvergenceWarning: Maximum number of iterations reached

**Warning:**
```
ConvergenceWarning: lbfgs failed to converge (status=1)
```

**Solution:**
```python
# Option 1: Increase max_iter
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)  # Default is 100

# Option 2: Scale features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Option 3: Try different solver
model = LogisticRegression(solver='saga', max_iter=1000)

# Option 4: Adjust tolerance
model = LogisticRegression(tol=1e-3, max_iter=1000)  # Default tol=1e-4
```

## Categorical Data Errors

### ValueError: could not convert string to float

**Error when trying to use categorical data in model:**

**Solution:**
```python
# Check for categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
print(f"Categorical columns: {categorical_cols.tolist()}")

# Option 1: Label Encoding
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for col in categorical_cols:
    df[f'{col}_encoded'] = le.fit_transform(df[col])

# Option 2: One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Option 3: Target Encoding
from sklearn.preprocessing import TargetEncoder

encoder = TargetEncoder()
df[categorical_cols] = encoder.fit_transform(df[categorical_cols], df['target'])
```

## Random State Issues

### Problem: Results not reproducible

**Solution:**
```python
# Set all random seeds at the start of notebook
import random
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Set random_state in all sklearn functions
X_train, X_test = train_test_split(X, y, random_state=SEED)
model = RandomForestClassifier(random_state=SEED)

# For TensorFlow/Keras
import tensorflow as tf
tf.random.set_seed(SEED)

# For PyTorch
import torch
torch.manual_seed(SEED)
```

## Plotting Errors

### UserWarning: matplotlib is currently using agg, a non-GUI backend

**Solution:**
```python
# For Jupyter notebooks
%matplotlib inline

# Or use different backend
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
```

### ValueError: x and y must be the same size

**Error in plotting:**

**Solution:**
```python
# Check array lengths
print(f"x length: {len(x)}")
print(f"y length: {len(y)}")

# Ensure same length
assert len(x) == len(y), "Arrays must have same length"

# Common cause: index mismatch
# Reset indices
df = df.reset_index(drop=True)
```

## Performance Issues

### Problem: Model training is very slow

**Solutions:**

```python
# 1. Use fewer estimators (for ensemble methods)
model = RandomForestClassifier(n_estimators=50)  # Instead of 1000

# 2. Limit max_depth
model = RandomForestClassifier(max_depth=10)

# 3. Use n_jobs for parallel processing
model = RandomForestClassifier(n_jobs=-1)  # Use all cores

# 4. Sample data for development
df_sample = df.sample(frac=0.1, random_state=42)

# 5. Use faster algorithms for prototyping
from sklearn.linear_model import SGDClassifier
model = SGDClassifier()  # Much faster than SVC

# 6. Reduce features
from sklearn.feature_selection import SelectKBest
selector = SelectKBest(k=20)
X_reduced = selector.fit_transform(X_train, y_train)
```

## Version Compatibility Issues

### AttributeError: 'DataFrame' object has no attribute 'append'

**Error in newer pandas versions (>= 2.0):**

**Solution:**
```python
# WRONG (deprecated in pandas 2.0)
df = df.append(new_row)

# CORRECT
df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
```

### FutureWarning or DeprecationWarning

**Solution:**
```python
# Update packages
pip install --upgrade scikit-learn pandas numpy matplotlib

# Check versions
import sklearn
import pandas as pd
import numpy as np

print(f"sklearn version: {sklearn.__version__}")
print(f"pandas version: {pd.__version__}")
print(f"numpy version: {np.__version__}")

# Suppress warnings temporarily (not recommended for production)
import warnings
warnings.filterwarnings('ignore')
```

## Debugging Tips

### General Debugging Strategy

```python
# 1. Check data at each step
print("Step 1: Data shape", df.shape)
print("Step 2: After cleaning", df_clean.shape)
print("Step 3: Missing values", df_clean.isnull().sum().sum())

# 2. Use assert statements
assert df.shape[0] > 0, "DataFrame is empty!"
assert 'target' in df.columns, "Target column missing!"
assert df.isnull().sum().sum() == 0, "Still have missing values!"

# 3. Inspect intermediate results
print(df.head())
print(df.dtypes)
print(df.describe())

# 4. Check for edge cases
print(f"Min value: {df['column'].min()}")
print(f"Max value: {df['column'].max()}")
print(f"Unique values: {df['column'].nunique()}")

# 5. Use try-except for graceful handling
try:
    model.fit(X_train, y_train)
except Exception as e:
    print(f"Error: {e}")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"Data types: {X_train.dtypes}")
```

### Systematic Error Resolution

When you encounter an error:

1. **Read the error message carefully** - It usually tells you exactly what's wrong
2. **Check the line number** - Go to that specific line in your code
3. **Print intermediate values** - See what the data looks like at that point
4. **Check data types** - `df.dtypes`, `type(variable)`
5. **Check shapes** - `df.shape`, `array.shape`
6. **Check for NaN/inf** - `df.isnull().sum()`, `np.isinf(df).sum()`
7. **Simplify** - Comment out complex parts and test basic functionality first
8. **Search the error** - Copy the error message and search online
9. **Check documentation** - Read the function/method documentation
10. **Ask for help** - Provide error message, code snippet, and what you've tried

## Quick Diagnostic Commands

```python
# Quick dataset health check
def diagnose_dataset(df):
    """Quick diagnostic of dataset health."""
    print("=" * 60)
    print("DATASET DIAGNOSTIC")
    print("=" * 60)
    
    print(f"\n1. SHAPE: {df.shape[0]} rows × {df.shape[1]} columns")
    
    print(f"\n2. MEMORY USAGE: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print(f"\n3. MISSING VALUES:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("   ✓ No missing values")
    else:
        print(missing[missing > 0])
    
    print(f"\n4. DUPLICATES: {df.duplicated().sum()}")
    
    print(f"\n5. DATA TYPES:")
    print(df.dtypes.value_counts())
    
    print(f"\n6. NUMERIC COLUMNS:")
    numeric = df.select_dtypes(include=[np.number]).columns
    print(f"   {len(numeric)} numeric columns")
    for col in numeric:
        print(f"   - {col}: min={df[col].min():.2f}, max={df[col].max():.2f}")
    
    print(f"\n7. CATEGORICAL COLUMNS:")
    categorical = df.select_dtypes(include=['object']).columns
    print(f"   {len(categorical)} categorical columns")
    for col in categorical:
        print(f"   - {col}: {df[col].nunique()} unique values")
    
    print("\n" + "=" * 60)

# Usage
diagnose_dataset(df)
```

Remember: Most errors are simple issues like typos, wrong column names, shape mismatches, or missing data handling. Always check the basics first!
