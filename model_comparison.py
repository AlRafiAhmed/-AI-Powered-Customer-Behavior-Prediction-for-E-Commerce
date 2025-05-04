import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report
import joblib

# Load and preprocess data
df = pd.read_csv('ecommerce_customer_data_custom_ratios.csv')
df_clean = df.dropna()

# Remove outliers
def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return data[(data[column] >= lower) & (data[column] <= upper)]

numeric_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns.drop(['Customer ID', 'Churn'])
df_outliers_removed = df_clean.copy()
for col in numeric_cols:
    df_outliers_removed = remove_outliers_iqr(df_outliers_removed, col)

# Preprocess
df_model = df_outliers_removed.drop(columns=['Customer ID', 'Purchase Date', 'Customer Name'])
le = LabelEncoder()
for col in df_model.select_dtypes(include='object').columns:
    df_model[col] = le.fit_transform(df_model[col])

# Stratified sample
df_sampled = df_model.groupby('Churn', group_keys=False).apply(lambda x: x.sample(frac=0.1, random_state=42))

# Split data
X = df_sampled.drop(columns='Churn')
y = df_sampled['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Create a dictionary to store all models
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'Bagging': BaggingClassifier(base_estimator=DecisionTreeClassifier(), random_state=42),
    'Ensemble': VotingClassifier([
        ('rf', RandomForestClassifier(random_state=42)),
        ('knn', KNeighborsClassifier()),
        ('dt', DecisionTreeClassifier(random_state=42))
    ])
}

# Train and evaluate all models
results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'predictions': y_pred
    }
    print(f"{name} Accuracy: {accuracy:.4f}")
    print(f"{name} Classification Report:\n{classification_report(y_test, y_pred)}")

# Create visualization of model accuracies
plt.figure(figsize=(12, 6))
accuracies = [results[model]['accuracy'] for model in models.keys()]
plt.bar(models.keys(), accuracies)
plt.title('Model Accuracy Comparison')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('model_accuracies.png')
plt.close()

# Create confusion matrices for each model
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for idx, (name, result) in enumerate(results.items()):
    cm = confusion_matrix(y_test, result['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[idx])
    axes[idx].set_title(f'{name} Confusion Matrix')
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('confusion_matrices.png')
plt.close()

# Create ROC curves for each model
plt.figure(figsize=(10, 8))
for name, result in results.items():
    y_pred_proba = result['model'].predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for All Models')
plt.legend()
plt.savefig('roc_curves.png')
plt.close()

# Find the best model
best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
best_model = results[best_model_name]['model']
print(f"\nBest performing model: {best_model_name}")

# Save the best model
joblib.dump(best_model, 'best_customer_behavior_model.joblib')
print("Best model saved as 'best_customer_behavior_model.joblib'")

# Save all models
for name, result in results.items():
    joblib.dump(result['model'], f'{name.lower().replace(" ", "_")}_model.joblib')
print("\nAll models have been saved successfully!") 