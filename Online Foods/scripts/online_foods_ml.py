import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, precision_recall_curve
from sklearn.metrics import ConfusionMatrixDisplay, auc
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import warnings
import os
import time
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

# Create output directories if they don't exist
if not os.path.exists('output/visualizations'):
    os.makedirs('output/visualizations')
if not os.path.exists('output/models'):
    os.makedirs('output/models')

# Set style for matplotlib plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

# Load the data
print("Loading data...")
df = pd.read_csv('data/onlinefoods.csv')

# Clean the data
# Remove trailing commas from column names if present
df.columns = df.columns.str.rstrip(',')

# Convert 'Output' to boolean
df['Output'] = df['Output'].map({'Yes': True, 'No': False})

# If the last column is unnamed and contains Yes/No, clean it
if df.columns[-1].startswith('Unnamed') or df.columns[-1] == '':
    # Rename it to something meaningful
    df.rename(columns={df.columns[-1]: 'Uses_Online_Food'}, inplace=True)
    # Convert to boolean
    df['Uses_Online_Food'] = df['Uses_Online_Food'].map({'Yes': True, 'No': False})

# Display basic information
print("\nDataset Overview:")
print(f"Shape: {df.shape}")
print("\nTarget variable distribution:")
print(df['Output'].value_counts(normalize=True) * 100)

# Define features and target
X = df.drop(['Output', 'Feedback'], axis=1)
if 'Uses_Online_Food' in X.columns:
    X = X.drop('Uses_Online_Food', axis=1)
y = df['Output']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\nTraining set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# Define preprocessing for numerical and categorical features
numerical_features = ['Age', 'Family size', 'latitude', 'longitude', 'Pin code']
categorical_features = ['Gender', 'Marital Status', 'Occupation', 'Monthly Income', 'Educational Qualifications']

# Create preprocessors
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessors
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create and evaluate multiple models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# Function to evaluate model
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'output/visualizations/confusion_matrix_{model_name.replace(" ", "_").lower()}.png')
    plt.close()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    plt.savefig(f'output/visualizations/roc_curve_{model_name.replace(" ", "_").lower()}.png')
    plt.close()
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }

# Train and evaluate models
results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Create pipeline with preprocessing and model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Evaluate the model
    results[name] = evaluate_model(pipeline, X_test, y_test, name)
    
    # Save the model
    joblib.dump(pipeline, f'output/models/model_{name.replace(" ", "_").lower()}.pkl')

# Compare models
plt.figure(figsize=(12, 8))
metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
for i, metric in enumerate(metrics):
    plt.subplot(2, 3, i+1)
    values = [results[model][metric] for model in models.keys()]
    sns.barplot(x=list(models.keys()), y=values)
    plt.title(f'{metric.capitalize()} Comparison')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('output/visualizations/model_comparison.png')
plt.close()

# Feature importance for Random Forest
if 'Random Forest' in models:
    # Get the trained Random Forest model from the pipeline
    rf_pipeline = joblib.load('output/models/model_random_forest.pkl')
    rf_model = rf_pipeline.named_steps['model']
    
    # Get feature names after preprocessing
    preprocessor = rf_pipeline.named_steps['preprocessor']
    cat_features = preprocessor.transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_features)
    feature_names = numerical_features + list(cat_features)
    
    # Get feature importances
    importances = rf_model.feature_importances_
    
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    
    # Plot feature importances
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importances - Random Forest')
    plt.bar(range(min(20, len(importances))), importances[indices[:20]], align='center')
    plt.xticks(range(min(20, len(importances))), [feature_names[i] for i in indices[:20]], rotation=90)
    plt.tight_layout()
    plt.savefig('output/visualizations/feature_importances.png')
    plt.close()

# Hyperparameter tuning for the best model
best_model_name = max(results, key=lambda x: results[x]['f1'])
print(f"\nPerforming hyperparameter tuning for {best_model_name}...")

if best_model_name == 'Logistic Regression':
    param_grid = {
        'model__C': [0.01, 0.1, 1, 10, 100],
        'model__solver': ['liblinear', 'saga']
    }
elif best_model_name == 'Random Forest':
    param_grid = {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [None, 10, 20, 30],
        'model__min_samples_split': [2, 5, 10]
    }
else:  # Gradient Boosting
    param_grid = {
        'model__n_estimators': [50, 100, 200],
        'model__learning_rate': [0.01, 0.1, 0.2],
        'model__max_depth': [3, 5, 7]
    }

# Create pipeline with preprocessing and model
best_model = models[best_model_name]
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', best_model)
])

# Perform grid search
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Evaluate the tuned model
tuned_model = grid_search.best_estimator_
tuned_results = evaluate_model(tuned_model, X_test, y_test, f"Tuned {best_model_name}")

# Save the tuned model
joblib.dump(tuned_model, 'output/models/model_tuned.pkl')

# Compare original vs tuned model
plt.figure(figsize=(10, 6))
metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
x = np.arange(len(metrics))
width = 0.35

plt.bar(x - width/2, [results[best_model_name][m] for m in metrics], width, label=f'Original {best_model_name}')
plt.bar(x + width/2, [tuned_results[m] for m in metrics], width, label=f'Tuned {best_model_name}')

plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title(f'Original vs Tuned {best_model_name}')
plt.xticks(x, metrics)
plt.legend()
plt.ylim(0, 1)
plt.savefig('output/visualizations/tuned_model_comparison.png')
plt.close()

print("\nMachine Learning Analysis completed successfully!")
print("Check the generated image files for visualizations in the output directory.") 