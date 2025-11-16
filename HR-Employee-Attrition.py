import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================================
# STEP 1: Load Data
# =====================================================================
df = pd.read_excel('WA_Fn-UseC_-HR-Employee-Attrition.xlsx')

# =====================================================================
# STEP 2: Basic Exploration
# =====================================================================
print("Shape of data:", df.shape)
print("\nInfo:")
print(df.info())
print("\nDescribe (all columns):")
print(df.describe(include='all'))

# Quick data quality check (print once)
print("\nNull values per column:")
print(df.isnull().sum())
print("\nNumber of duplicated rows:", df.duplicated().sum())

# Create target flag once
df['AttritionFlag'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# Overall attrition rate
overall_rate = df['AttritionFlag'].mean() * 100
print(f"\nOverall Attrition Rate: {overall_rate:.1f}%")

# =====================================================================
# STEP 3: Key KPIs – Who tends to leave?
# =====================================================================
for col in ['OverTime', 'Department', 'JobRole', 'BusinessTravel', 'MaritalStatus']:
    print(f"\nAttrition rate by {col} (%):")
    print(
        (df.groupby(col)['AttritionFlag']
           .mean()
           .mul(100)
           .round(1)
           .sort_values(ascending=False))
    )

# =====================================================================
# STEP 4: Numeric Differences by Attrition
# =====================================================================
for col in ['MonthlyIncome', 'YearsAtCompany', 'DistanceFromHome', 'Age', 'JobSatisfaction']:
    print(f"\n{col} by Attrition:")
    print(
        df.groupby('Attrition')[col]
          .agg(['mean', 'median', 'std'])
          .round(2)
    )

# =====================================================================
# STEP 5: Visualisations (focused, not everything)
# =====================================================================

# 5.1 Distributions of key numeric features
key_numeric = ['MonthlyIncome', 'YearsAtCompany', 'DistanceFromHome', 'Age']
for name in key_numeric:
    plt.figure()
    sns.histplot(df[name], bins=30, kde=True)
    plt.title(f"{name} distribution")
    plt.xlabel(name)
    plt.ylabel("Count")
    plt.show()

# 5.2 Boxplots: numeric vs Attrition
for name in key_numeric + ['JobSatisfaction']:
    plt.figure()
    sns.boxplot(data=df, x='Attrition', y=name)
    plt.title(f"{name} vs Attrition")
    plt.show()

# 5.3 Satisfaction-related factors vs Attrition (as bar plots of attrition rate)
plt.figure()
(df.groupby('EnvironmentSatisfaction')['AttritionFlag'].mean() * 100).round(1).plot(kind="bar")
plt.title("Attrition rate by EnvironmentSatisfaction")
plt.ylabel("Attrition rate (%)")
plt.show()

plt.figure()
(df.groupby('JobInvolvement')['AttritionFlag'].mean() * 100).round(1).plot(kind="bar")
plt.title("Attrition rate by JobInvolvement")
plt.ylabel("Attrition rate (%)")
plt.show()

plt.figure()
(df.groupby('JobSatisfaction')['AttritionFlag'].mean() * 100).round(1).plot(kind="bar")
plt.title("Attrition rate by JobSatisfaction")
plt.ylabel("Attrition rate (%)")
plt.show()

plt.figure()
(df.groupby('YearsSinceLastPromotion')['AttritionFlag'].mean() * 100).round(1).plot(kind="bar")
plt.title("Attrition rate by YearsSinceLastPromotion")
plt.ylabel("Attrition rate (%)")
plt.show()

plt.figure()
(df.groupby('YearsWithCurrManager')['AttritionFlag'].mean() * 100).round(1).plot(kind="bar")
plt.title("Attrition rate by YearsWithCurrManager")
plt.ylabel("Attrition rate (%)")
plt.show()

# JobSatisfaction by Department (pattern across departments)
print("\nAverage JobSatisfaction by Department:")
print(df.groupby('Department')['JobSatisfaction'].mean().sort_values().round(2))

plt.figure()
(df.groupby('Department')['JobSatisfaction'].mean().round(2)).plot(kind='bar')
plt.title("Average JobSatisfaction by Department")
plt.ylabel("Average JobSatisfaction")
plt.show()

# =====================================================================
# STEP 6: Statistical Analysis – Career & Manager Factors
# =====================================================================

print("\nAttrition by YearsSinceLastPromotion (%):")
print((df.groupby('YearsSinceLastPromotion')['AttritionFlag'].mean() * 100).round(1))

print("\nAttrition by YearsWithCurrManager (%):")
print((df.groupby('YearsWithCurrManager')['AttritionFlag'].mean() * 100).round(1))

print("\nCorrelation with AttritionFlag (numeric features):")
print(
    df.select_dtypes(include=[np.number])
      .corr()['AttritionFlag']
      .sort_values(ascending=False)
      .round(3)
)
