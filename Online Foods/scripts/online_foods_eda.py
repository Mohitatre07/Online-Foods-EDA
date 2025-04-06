import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium.plugins import HeatMap
import warnings
import os
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import json
from matplotlib.colors import LinearSegmentedColormap

# Suppress warnings
warnings.filterwarnings('ignore')

# Set style for matplotlib plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

# Create output directory if it doesn't exist
if not os.path.exists('output/visualizations'):
    os.makedirs('output/visualizations')

# Load the data
df = pd.read_csv('data/onlinefoods.csv')

# Display basic information about the dataset
print("Dataset Overview:")
print(f"Shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Data types
print("\nData types:")
print(df.dtypes)

# Basic statistics
print("\nBasic statistics:")
print(df.describe(include='all'))

# Clean the data
# Remove trailing commas from column names if present
df.columns = df.columns.str.rstrip(',')

# Check unique values in categorical columns
categorical_cols = ['Gender', 'Marital Status', 'Occupation', 'Monthly Income', 
                    'Educational Qualifications', 'Output', 'Feedback']

print("\nUnique values in categorical columns:")
for col in categorical_cols:
    print(f"\n{col}: {df[col].unique()}")

# Convert 'Output' and last column to proper boolean if needed
if 'Output' in df.columns:
    df['Output'] = df['Output'].map({'Yes': True, 'No': False})

# If the last column is unnamed and contains Yes/No, clean it
if df.columns[-1].startswith('Unnamed') or df.columns[-1] == '':
    # Rename it to something meaningful
    df.rename(columns={df.columns[-1]: 'Uses_Online_Food'}, inplace=True)
    # Convert to boolean
    df['Uses_Online_Food'] = df['Uses_Online_Food'].map({'Yes': True, 'No': False})

# Age distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], kde=True, bins=20)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.savefig('output/visualizations/age_distribution.png')
plt.close()

# Gender distribution
plt.figure(figsize=(8, 6))
gender_counts = df['Gender'].value_counts()
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Gender Distribution')
plt.axis('equal')
plt.savefig('output/visualizations/gender_distribution.png')
plt.close()

# Marital Status distribution
plt.figure(figsize=(10, 6))
marital_counts = df['Marital Status'].value_counts()
sns.barplot(x=marital_counts.index, y=marital_counts.values)
plt.title('Marital Status Distribution')
plt.xlabel('Marital Status')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.savefig('output/visualizations/marital_status_distribution.png')
plt.close()

# Occupation distribution
plt.figure(figsize=(12, 6))
occupation_counts = df['Occupation'].value_counts()
sns.barplot(x=occupation_counts.index, y=occupation_counts.values)
plt.title('Occupation Distribution')
plt.xlabel('Occupation')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.savefig('output/visualizations/occupation_distribution.png')
plt.close()

# Monthly Income distribution
plt.figure(figsize=(12, 6))
income_counts = df['Monthly Income'].value_counts()
sns.barplot(x=income_counts.index, y=income_counts.values)
plt.title('Monthly Income Distribution')
plt.xlabel('Monthly Income')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.savefig('output/visualizations/income_distribution.png')
plt.close()

# Educational Qualifications distribution
plt.figure(figsize=(12, 6))
education_counts = df['Educational Qualifications'].value_counts()
sns.barplot(x=education_counts.index, y=education_counts.values)
plt.title('Educational Qualifications Distribution')
plt.xlabel('Educational Qualifications')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.savefig('output/visualizations/education_distribution.png')
plt.close()

# Family size distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='Family size', data=df)
plt.title('Family Size Distribution')
plt.xlabel('Family Size')
plt.ylabel('Count')
plt.savefig('output/visualizations/family_size_distribution.png')
plt.close()

# Output distribution
plt.figure(figsize=(8, 6))
output_counts = df['Output'].value_counts()
plt.pie(output_counts, labels=['Yes', 'No'], autopct='%1.1f%%', startangle=90)
plt.title('Online Food Service Usage')
plt.axis('equal')
plt.savefig('output/visualizations/output_distribution.png')
plt.close()

# Feedback distribution
plt.figure(figsize=(8, 6))
feedback_counts = df['Feedback'].value_counts()
plt.pie(feedback_counts, labels=feedback_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Feedback Distribution')
plt.axis('equal')
plt.savefig('output/visualizations/feedback_distribution.png')
plt.close()

# Relationship between Age and Output
plt.figure(figsize=(10, 6))
sns.boxplot(x='Output', y='Age', data=df)
plt.title('Age vs Online Food Service Usage')
plt.savefig('output/visualizations/age_vs_output.png')
plt.close()

# Relationship between Gender and Output
plt.figure(figsize=(10, 6))
gender_output = pd.crosstab(df['Gender'], df['Output'])
gender_output_pct = gender_output.div(gender_output.sum(axis=1), axis=0) * 100
gender_output_pct.plot(kind='bar', stacked=True)
plt.title('Gender vs Online Food Service Usage')
plt.xlabel('Gender')
plt.ylabel('Percentage')
plt.savefig('output/visualizations/gender_vs_output.png')
plt.close()

# Relationship between Marital Status and Output
plt.figure(figsize=(10, 6))
marital_output = pd.crosstab(df['Marital Status'], df['Output'])
marital_output_pct = marital_output.div(marital_output.sum(axis=1), axis=0) * 100
marital_output_pct.plot(kind='bar', stacked=True)
plt.title('Marital Status vs Online Food Service Usage')
plt.xlabel('Marital Status')
plt.ylabel('Percentage')
plt.savefig('output/visualizations/marital_status_vs_output.png')
plt.close()

# Relationship between Occupation and Output
plt.figure(figsize=(12, 6))
occupation_output = pd.crosstab(df['Occupation'], df['Output'])
occupation_output_pct = occupation_output.div(occupation_output.sum(axis=1), axis=0) * 100
occupation_output_pct.plot(kind='bar', stacked=True)
plt.title('Occupation vs Online Food Service Usage')
plt.xlabel('Occupation')
plt.ylabel('Percentage')
plt.xticks(rotation=45)
plt.savefig('output/visualizations/occupation_vs_output.png')
plt.close()

# Relationship between Monthly Income and Output
plt.figure(figsize=(12, 6))
income_output = pd.crosstab(df['Monthly Income'], df['Output'])
income_output_pct = income_output.div(income_output.sum(axis=1), axis=0) * 100
income_output_pct.plot(kind='bar', stacked=True)
plt.title('Monthly Income vs Online Food Service Usage')
plt.xlabel('Monthly Income')
plt.ylabel('Percentage')
plt.xticks(rotation=45)
plt.savefig('output/visualizations/income_vs_output.png')
plt.close()

# Relationship between Educational Qualifications and Output
plt.figure(figsize=(12, 6))
education_output = pd.crosstab(df['Educational Qualifications'], df['Output'])
education_output_pct = education_output.div(education_output.sum(axis=1), axis=0) * 100
education_output_pct.plot(kind='bar', stacked=True)
plt.title('Educational Qualifications vs Online Food Service Usage')
plt.xlabel('Educational Qualifications')
plt.ylabel('Percentage')
plt.xticks(rotation=45)
plt.savefig('output/visualizations/education_vs_output.png')
plt.close()

# Relationship between Family Size and Output
plt.figure(figsize=(10, 6))
family_output = pd.crosstab(df['Family size'], df['Output'])
family_output_pct = family_output.div(family_output.sum(axis=1), axis=0) * 100
family_output_pct.plot(kind='bar', stacked=True)
plt.title('Family Size vs Online Food Service Usage')
plt.xlabel('Family Size')
plt.ylabel('Percentage')
plt.savefig('output/visualizations/family_size_vs_output.png')
plt.close()

# Correlation heatmap for numerical variables
plt.figure(figsize=(10, 8))
numerical_cols = ['Age', 'Family size', 'latitude', 'longitude']
correlation = df[numerical_cols].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.savefig('output/visualizations/correlation_heatmap.png')
plt.close()

# Geographic analysis - Map visualization
# Create a map centered at the mean latitude and longitude
map_center = [df['latitude'].mean(), df['longitude'].mean()]
m = folium.Map(location=map_center, zoom_start=12)

# Add markers for each location
for idx, row in df.iterrows():
    popup_text = f"Age: {row['Age']}<br>Gender: {row['Gender']}<br>Occupation: {row['Occupation']}<br>Uses Online Food: {row['Output']}"
    color = 'green' if row['Output'] else 'red'
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=popup_text,
        icon=folium.Icon(color=color)
    ).add_to(m)

# Save the map
m.save('output/dashboards/online_food_map.html')

# Create a heatmap of online food service usage
heat_data = df[df['Output'] == True][['latitude', 'longitude']].values.tolist()
HeatMap(heat_data).add_to(folium.Map(location=map_center, zoom_start=12)).save('output/dashboards/online_food_heatmap.html')

# Advanced analysis with Plotly
# Age distribution by gender
fig = px.histogram(df, x='Age', color='Gender', marginal='box', 
                  title='Age Distribution by Gender')
fig.write_html('output/dashboards/age_distribution_by_gender.html')

# Online food service usage by age and gender
fig = px.scatter(df, x='Age', y='Family size', color='Output', 
                symbol='Gender', size='Age',
                title='Online Food Service Usage by Age, Family Size and Gender')
fig.write_html('output/dashboards/usage_by_age_family_gender.html')

# Sunburst chart for hierarchical view
fig = px.sunburst(df, path=['Gender', 'Marital Status', 'Output'], 
                 title='Hierarchical View of Online Food Service Usage')
fig.write_html('output/dashboards/hierarchical_view.html')

# 3D scatter plot with geographical data
fig = px.scatter_3d(df, x='latitude', y='longitude', z='Age',
                   color='Output', symbol='Gender',
                   title='3D Geographical Distribution by Age and Online Food Service Usage')
fig.write_html('output/dashboards/3d_geographical_distribution.html')

# Statistical tests
# Chi-square test for categorical variables vs Output
print("\nStatistical Tests:")
categorical_cols = ['Gender', 'Marital Status', 'Occupation', 'Monthly Income', 'Educational Qualifications']
for col in categorical_cols:
    contingency_table = pd.crosstab(df[col], df['Output'])
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"\nChi-square test for {col} vs Output:")
    print(f"Chi2 value: {chi2:.4f}")
    print(f"p-value: {p:.4f}")
    print(f"Significant relationship: {p < 0.05}")

# T-test for Age between Output groups
yes_ages = df[df['Output'] == True]['Age']
no_ages = df[df['Output'] == False]['Age']
t_stat, p_val = stats.ttest_ind(yes_ages, no_ages)
print("\nT-test for Age between Output groups:")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_val:.4f}")
print(f"Significant difference: {p_val < 0.05}")

# Create a dashboard with Plotly
dashboard = make_subplots(
    rows=2, cols=2,
    subplot_titles=("Age Distribution", "Gender Distribution", 
                    "Online Food Service Usage", "Feedback Distribution"),
    specs=[[{"type": "histogram"}, {"type": "pie"}],
           [{"type": "pie"}, {"type": "pie"}]]
)

# Age Distribution
dashboard.add_trace(
    go.Histogram(x=df['Age'], nbinsx=20, name="Age"),
    row=1, col=1
)

# Gender Distribution
dashboard.add_trace(
    go.Pie(labels=gender_counts.index, values=gender_counts.values, name="Gender"),
    row=1, col=2
)

# Output Distribution
dashboard.add_trace(
    go.Pie(labels=['Yes', 'No'], values=output_counts.values, name="Output"),
    row=2, col=1
)

# Feedback Distribution
dashboard.add_trace(
    go.Pie(labels=feedback_counts.index, values=feedback_counts.values, name="Feedback"),
    row=2, col=2
)

dashboard.update_layout(height=800, width=1000, title_text="Online Food Service Dashboard")
dashboard.write_html('output/dashboards/dashboard.html')

print("\nExploratory Data Analysis completed successfully!")
print("Check the generated image files and HTML files for visualizations in the output directory.") 