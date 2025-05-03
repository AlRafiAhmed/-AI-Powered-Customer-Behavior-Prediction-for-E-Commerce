import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Set style for better visualizations
plt.style.use('seaborn')
sns.set_palette("husl")

# 1. Load and explore the data
print("Loading data...")
df = pd.read_csv('ecommerce_customer_data_custom_ratios.csv')

# 2. Check for null values
print("\nChecking for null values:")
print(df.isnull().sum())

# Remove null values
df_clean = df.dropna()
print(f"\nShape after removing null values: {df_clean.shape}")

# 3. Visualize outliers
def plot_outliers(data, column):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=data[column])
    plt.title(f'Boxplot of {column}')
    plt.savefig(f'outliers_{column}.png')
    plt.close()

# Plot outliers for numeric columns
numeric_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns.drop(['Customer ID', 'Churn'])
for col in numeric_cols:
    plot_outliers(df_clean, col)

# 4. Remove outliers using IQR method
def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return data[(data[column] >= lower) & (data[column] <= upper)]

df_outliers_removed = df_clean.copy()
for col in numeric_cols:
    df_outliers_removed = remove_outliers_iqr(df_outliers_removed, col)

print(f"\nShape after removing outliers: {df_outliers_removed.shape}")

# 5. Create correlation matrix
plt.figure(figsize=(12, 8))
correlation_matrix = df_outliers_removed.select_dtypes(include=['float64', 'int64']).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

# 6. Prepare data for modeling
df_model = df_outliers_removed.drop(columns=['Customer ID', 'Purchase Date', 'Customer Name'])
le = LabelEncoder()
for col in df_model.select_dtypes(include='object').columns:
    df_model[col] = le.fit_transform(df_model[col])

# 7. Split the data
X = df_model.drop(columns='Churn')
y = df_model['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 8. Train and evaluate multiple models
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42)
}

results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"{name} Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'confusion_matrix_{name.lower().replace(" ", "_")}.png')
    plt.close()

# 9. Find the best model
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
print(f"\nBest performing model: {best_model_name} with accuracy: {results[best_model_name]:.4f}")

# 10. Save the best model and scaler
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\nModel and scaler have been saved as 'best_model.pkl' and 'scaler.pkl'") 