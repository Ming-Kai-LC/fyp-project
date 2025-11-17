# Exploratory Data Analysis (EDA) - Comprehensive Guide

This reference provides detailed guidance on systematic exploratory data analysis and visualization selection.

## EDA Three-Stage Approach

### Stage 1: Univariate Analysis
Examine each variable individually to understand distributions, central tendencies, and variability.

### Stage 2: Bivariate Analysis
Investigate relationships between pairs of variables, especially feature-target relationships.

### Stage 3: Multivariate Analysis
Explore interactions among multiple variables, correlation structures, and complex patterns.

## Visualization Selection by Data Type

### Numerical → Numerical

**Scatter Plot**
```python
plt.figure(figsize=(10, 6))
plt.scatter(df['feature1'], df['feature2'], alpha=0.5)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Relationship between Feature 1 and Feature 2')
plt.show()

# With seaborn for better styling
sns.scatterplot(data=df, x='feature1', y='feature2', hue='target')
plt.show()
```
- **Use when**: Examining relationship between two continuous variables
- **Shows**: Correlation, linearity, clusters, outliers

**Line Plot**
```python
plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['value'], marker='o')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Time Series Trend')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```
- **Use when**: Time series data, trends over time
- **Shows**: Temporal patterns, seasonality, trends

**Heatmap (Correlation Matrix)**
```python
# Compute correlation matrix
corr_matrix = df[numeric_cols].corr()

# Visualize
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()

# Focus on target correlations
target_corr = corr_matrix['target'].sort_values(ascending=False)
print("Correlations with target:")
print(target_corr)
```
- **Use when**: Understanding relationships among multiple numeric variables
- **Shows**: Multicollinearity, feature redundancy, strong associations

### Categorical → Numerical

**Box Plot**
```python
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='category', y='numeric_value')
plt.xticks(rotation=45)
plt.title('Distribution of Numeric Value by Category')
plt.tight_layout()
plt.show()

# With statistical annotations
from scipy import stats
sns.boxplot(data=df, x='category', y='numeric_value')
# Add mean markers
means = df.groupby('category')['numeric_value'].mean()
positions = range(len(means))
plt.plot(positions, means, 'r^', markersize=10, label='Mean')
plt.legend()
plt.show()
```
- **Use when**: Comparing numeric distributions across categories
- **Shows**: Median, quartiles, outliers, distribution differences

**Violin Plot**
```python
plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x='category', y='numeric_value', inner='box')
plt.xticks(rotation=45)
plt.title('Distribution Density by Category')
plt.tight_layout()
plt.show()
```
- **Use when**: Want to see both distribution shape and summary statistics
- **Shows**: Distribution density, median, quartiles

**Bar Plot (Aggregated Statistics)**
```python
# Mean by category
mean_by_cat = df.groupby('category')['numeric_value'].mean().sort_values()

plt.figure(figsize=(10, 6))
mean_by_cat.plot(kind='bar', color='skyblue', edgecolor='black')
plt.xlabel('Category')
plt.ylabel('Mean Numeric Value')
plt.title('Average Value by Category')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# With error bars (confidence intervals)
means = df.groupby('category')['numeric_value'].mean()
stds = df.groupby('category')['numeric_value'].std()
means.plot(kind='bar', yerr=stds, capsize=4)
plt.show()
```
- **Use when**: Comparing aggregated statistics across categories
- **Shows**: Means, sums, or other aggregations with optional error bars

### Categorical → Categorical

**Stacked Bar Chart**
```python
# Cross-tabulation
ct = pd.crosstab(df['category1'], df['category2'])

# Stacked bar
ct.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')
plt.xlabel('Category 1')
plt.ylabel('Count')
plt.title('Distribution of Category 2 within Category 1')
plt.legend(title='Category 2')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Normalized (proportions)
ct_normalized = ct.div(ct.sum(axis=1), axis=0)
ct_normalized.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.ylabel('Proportion')
plt.show()
```
- **Use when**: Examining relationship between two categorical variables
- **Shows**: Joint distributions, conditional probabilities

**Heatmap (Cross-tabulation)**
```python
ct = pd.crosstab(df['category1'], df['category2'])

plt.figure(figsize=(10, 8))
sns.heatmap(ct, annot=True, fmt='d', cmap='YlOrRd', linewidths=0.5)
plt.xlabel('Category 2')
plt.ylabel('Category 1')
plt.title('Frequency Heatmap')
plt.tight_layout()
plt.show()
```
- **Use when**: Visualizing frequency table
- **Shows**: Joint frequencies, patterns in categorical relationships

**Mosaic Plot** (requires statsmodels)
```python
from statsmodels.graphics.mosaicplot import mosaic

mosaic(df, ['category1', 'category2'], title='Mosaic Plot')
plt.show()
```
- **Use when**: Showing proportions and relationships
- **Shows**: Relative frequencies, associations

## Distribution Analysis

### Histogram
```python
plt.figure(figsize=(10, 6))
plt.hist(df['numeric_feature'], bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Distribution of Numeric Feature')
plt.axvline(df['numeric_feature'].mean(), color='red', linestyle='--', label='Mean')
plt.axvline(df['numeric_feature'].median(), color='green', linestyle='--', label='Median')
plt.legend()
plt.show()

# Multiple histograms
df[numeric_cols].hist(bins=30, figsize=(15, 10), edgecolor='black')
plt.tight_layout()
plt.show()
```
- **Use when**: Understanding distribution of continuous variable
- **Shows**: Shape, modality, skewness, outliers

### KDE (Kernel Density Estimate)
```python
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x='numeric_feature', fill=True)
plt.title('Density Distribution')
plt.show()

# Compare distributions across categories
sns.kdeplot(data=df, x='numeric_feature', hue='category', fill=True)
plt.title('Distribution Comparison by Category')
plt.show()
```
- **Use when**: Smooth distribution visualization, comparing distributions
- **Shows**: Density, modality, overlapping distributions

### Q-Q Plot (Quantile-Quantile)
```python
from scipy import stats

stats.probplot(df['numeric_feature'], dist="norm", plot=plt)
plt.title('Q-Q Plot: Normal Distribution Check')
plt.show()
```
- **Use when**: Testing if data follows specific distribution (usually normal)
- **Shows**: Deviations from theoretical distribution

## Advanced Multivariate Visualizations

### Pair Plot
```python
# Select important features
important_features = ['feature1', 'feature2', 'feature3', 'target']

sns.pairplot(df[important_features], hue='target', diag_kind='kde', 
             plot_kws={'alpha': 0.6}, corner=True)
plt.show()
```
- **Use when**: Exploring relationships among multiple numeric variables
- **Shows**: Pairwise scatter plots, distributions, correlations by target

### Parallel Coordinates
```python
from pandas.plotting import parallel_coordinates

# Standardize data for better visualization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])

plt.figure(figsize=(12, 6))
parallel_coordinates(df_scaled[numeric_cols + ['target']], 'target', 
                     colormap='viridis', alpha=0.5)
plt.title('Parallel Coordinates Plot')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```
- **Use when**: Visualizing high-dimensional data, comparing classes
- **Shows**: Patterns across multiple dimensions, class separability

### 3D Scatter Plot
```python
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(df['feature1'], df['feature2'], df['feature3'], 
                     c=df['target'], cmap='viridis', alpha=0.6)
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
plt.colorbar(scatter, label='Target')
plt.title('3D Feature Space')
plt.show()
```
- **Use when**: Exploring relationships among three continuous variables
- **Shows**: 3D patterns, clusters, separability

## Statistical Summary Visualizations

### Distribution Summary Plot
```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Histogram
axes[0].hist(df['feature'], bins=30, edgecolor='black', alpha=0.7)
axes[0].set_title('Histogram')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')

# Box plot
axes[1].boxplot(df['feature'])
axes[1].set_title('Box Plot')
axes[1].set_ylabel('Value')

# KDE
sns.kdeplot(data=df, x='feature', fill=True, ax=axes[2])
axes[2].set_title('Density Plot')
axes[2].set_xlabel('Value')

plt.tight_layout()
plt.show()
```

### Statistical Test Visualization
```python
from scipy.stats import ttest_ind

# Compare two groups
group1 = df[df['category'] == 'A']['value']
group2 = df[df['category'] == 'B']['value']

t_stat, p_value = ttest_ind(group1, group2)

# Visualize with annotations
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='category', y='value')
plt.title(f'Group Comparison (p-value: {p_value:.4f})')

# Add significance indicator
if p_value < 0.05:
    plt.text(0.5, plt.ylim()[1] * 0.95, '***', ha='center', fontsize=20)
    
plt.show()
```

## Time Series Specific Visualizations

### Trend, Seasonality, Residuals
```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Decompose time series
decomposition = seasonal_decompose(df['value'], model='additive', period=12)

fig, axes = plt.subplots(4, 1, figsize=(12, 10))

df['value'].plot(ax=axes[0], title='Original')
decomposition.trend.plot(ax=axes[1], title='Trend')
decomposition.seasonal.plot(ax=axes[2], title='Seasonality')
decomposition.resid.plot(ax=axes[3], title='Residuals')

plt.tight_layout()
plt.show()
```

### Autocorrelation Plot
```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

plot_acf(df['value'], lags=40, ax=axes[0])
axes[0].set_title('Autocorrelation Function (ACF)')

plot_pacf(df['value'], lags=40, ax=axes[1])
axes[1].set_title('Partial Autocorrelation Function (PACF)')

plt.tight_layout()
plt.show()
```

## Interactive Visualizations (Optional)

### Using Plotly for Interactive Plots
```python
import plotly.express as px

# Interactive scatter
fig = px.scatter(df, x='feature1', y='feature2', color='target', 
                 hover_data=['feature3', 'feature4'],
                 title='Interactive Scatter Plot')
fig.show()

# Interactive 3D scatter
fig = px.scatter_3d(df, x='feature1', y='feature2', z='feature3', 
                    color='target', title='Interactive 3D Scatter')
fig.show()
```

## Publication-Quality Figure Settings

### Matplotlib RC Parameters
```python
import matplotlib as mpl

# Set publication-quality parameters
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.size'] = 12
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['xtick.labelsize'] = 11
mpl.rcParams['ytick.labelsize'] = 11
mpl.rcParams['legend.fontsize'] = 11
mpl.rcParams['figure.titlesize'] = 18

# For saving figures
plt.savefig('figure.png', dpi=300, bbox_inches='tight', transparent=False)
```

## EDA Systematic Workflow

### Step-by-Step Process
```python
# 1. Load and inspect
df = pd.read_csv('data.csv')
print(df.info())
print(df.describe())

# 2. Missing values visualization
import missingno as msno
msno.matrix(df)
plt.show()

# 3. Univariate analysis - distributions
df.hist(bins=30, figsize=(15, 10))
plt.tight_layout()
plt.show()

# 4. Univariate analysis - categorical
for col in df.select_dtypes(include='object').columns:
    print(f"\n{col}:")
    print(df[col].value_counts())

# 5. Bivariate analysis - correlations
corr_matrix = df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.show()

# 6. Bivariate analysis - target relationships
if 'target' in df.columns:
    target_corr = df.corr()['target'].sort_values(ascending=False)
    print("\nTop correlations with target:")
    print(target_corr.head(10))
    
    # Visualize top features
    top_features = target_corr.index[1:6]  # Top 5 features
    for feature in top_features:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x=feature, y='target')
        plt.title(f'{feature} vs Target (correlation: {target_corr[feature]:.3f})')
        plt.show()

# 7. Multivariate analysis
important_features = target_corr.index[1:6].tolist() + ['target']
sns.pairplot(df[important_features], hue='target', corner=True)
plt.show()

# 8. Outlier detection visualization
numeric_cols = df.select_dtypes(include=[np.number]).columns
fig, axes = plt.subplots(1, len(numeric_cols), figsize=(20, 5))
for i, col in enumerate(numeric_cols):
    axes[i].boxplot(df[col].dropna())
    axes[i].set_title(col)
    axes[i].set_ylabel('Value')
plt.tight_layout()
plt.show()
```

## Common Patterns to Look For

### Distribution Patterns
- **Normal**: Bell-shaped, symmetric
- **Skewed**: Asymmetric, tail on one side
- **Bimodal**: Two peaks, possible distinct groups
- **Uniform**: Flat, equal probability across range

### Relationship Patterns
- **Linear**: Straight line pattern
- **Non-linear**: Curved relationships
- **No relationship**: Random scatter
- **Clusters**: Distinct groupings
- **Outliers**: Points far from main pattern

### Temporal Patterns
- **Trend**: Long-term increase/decrease
- **Seasonality**: Regular periodic fluctuations
- **Cyclic**: Non-fixed period fluctuations
- **Irregular**: Random variations

## EDA Documentation Template

Document findings systematically:
```markdown
## EDA Findings

### Data Overview
- Samples: [n]
- Features: [n]
- Target variable: [name]
- Missing data: [percentage]

### Key Insights
1. **Feature X**: Normally distributed, strong positive correlation with target (r=0.75)
2. **Feature Y**: Right-skewed, contains outliers (>3 std from mean)
3. **Feature Z**: Categorical with imbalance (Class A: 80%, Class B: 20%)

### Relationships
- Strong positive correlation between X and target
- Non-linear relationship between Y and target
- Category A shows significantly higher target values

### Recommendations
1. Apply log transformation to Feature Y (skewed)
2. Handle class imbalance in Feature Z
3. Consider polynomial features for Y-target relationship
4. Remove outliers in Feature Y (confirmed data errors)
```

## Visualization Checklist

Before finalizing visualizations:
- [ ] Clear, descriptive titles
- [ ] Labeled axes with units
- [ ] Legend when multiple categories/series
- [ ] Appropriate color scheme (colorblind-friendly if possible)
- [ ] Readable font sizes (especially for presentations/papers)
- [ ] High resolution (300+ dpi for publications)
- [ ] Proper figure size for medium (screen vs. paper)
- [ ] Remove clutter (unnecessary grid lines, decorations)
- [ ] Consistent style across all figures
- [ ] Caption explaining what the figure shows
