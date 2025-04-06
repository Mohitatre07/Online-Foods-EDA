import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from matplotlib.colors import LinearSegmentedColormap

# Suppress warnings
warnings.filterwarnings('ignore')

# Set style for matplotlib plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

print("=" * 80)
print("Online Foods Dataset - Quick Start")
print("=" * 80)

# Check if the CSV file exists
if not os.path.exists('data/onlinefoods.csv'):
    print("Error: onlinefoods.csv file not found!")
    print("Please make sure the CSV file is in the data directory.")
    exit(1)

# Create output directory if it doesn't exist
if not os.path.exists('output/quick_start_viz'):
    os.makedirs('output/quick_start_viz')

# Load the data
print("\nLoading data...")
df = pd.read_csv('data/onlinefoods.csv')

# Clean the data
# Remove trailing commas from column names if present
df.columns = df.columns.str.rstrip(',')

# Convert 'Output' to boolean
df['Output'] = df['Output'].map({'Yes': True, 'No': False})

# Display basic information
print("\nDataset Overview:")
print(f"Shape: {df.shape}")
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")

# Display column names
print("\nColumns in the dataset:")
for col in df.columns:
    print(f"- {col}")

# Display basic statistics
print("\nBasic statistics for numerical columns:")
print(df.describe())

# Display value counts for categorical columns
categorical_cols = ['Gender', 'Marital Status', 'Occupation', 'Monthly Income', 
                   'Educational Qualifications', 'Output', 'Feedback']

print("\nValue counts for categorical columns:")
for col in categorical_cols:
    print(f"\n{col}:")
    print(df[col].value_counts())

# Create some basic visualizations
print("\nCreating basic visualizations...")

# 1. Age distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], kde=True, bins=15)
plt.title('Age Distribution')
plt.savefig('output/quick_start_viz/age_distribution.png')
plt.close()

# 2. Gender distribution
plt.figure(figsize=(8, 6))
df['Gender'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Gender Distribution')
plt.ylabel('')
plt.savefig('output/quick_start_viz/gender_distribution.png')
plt.close()

# 3. Online food service usage
plt.figure(figsize=(8, 6))
df['Output'].value_counts().plot(kind='pie', autopct='%1.1f%%', labels=['Yes', 'No'])
plt.title('Online Food Service Usage')
plt.ylabel('')
plt.savefig('output/quick_start_viz/online_food_usage.png')
plt.close()

# 4. Age vs Online food service usage
plt.figure(figsize=(10, 6))
sns.boxplot(x='Output', y='Age', data=df)
plt.title('Age vs Online Food Service Usage')
plt.savefig('output/quick_start_viz/age_vs_usage.png')
plt.close()

# 5. Occupation distribution
plt.figure(figsize=(12, 6))
df['Occupation'].value_counts().plot(kind='bar')
plt.title('Occupation Distribution')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('output/quick_start_viz/occupation_distribution.png')
plt.close()

print("\nQuick start visualizations created in the 'output/quick_start_viz' directory.")
print("\nTo run the full analysis, use:")
print("1. python scripts/setup.py       # Install dependencies")
print("2. python scripts/run_analysis.py # Run the complete analysis")
print("\nOr explore individual components:")
print("- python scripts/online_foods_eda.py       # Basic EDA")
print("- python scripts/online_foods_ml.py        # Machine Learning Analysis")
print("- python scripts/online_foods_dashboard.py # Interactive Dashboard")
print("=" * 80) 