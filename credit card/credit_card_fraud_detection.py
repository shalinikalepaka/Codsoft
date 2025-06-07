import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def generate_synthetic_data(n_samples=1000, fraud_ratio=0.1):
    """Generate synthetic credit card transaction data."""
    print("Generating synthetic data...")
    
    # Number of fraudulent transactions
    n_fraud = int(n_samples * fraud_ratio)
    n_normal = n_samples - n_fraud
    
    # Generate features
    n_features = 10
    X = np.random.randn(n_samples, n_features)
    
    # Generate labels (0 for normal, 1 for fraud)
    y = np.zeros(n_samples)
    y[:n_fraud] = 1
    
    # Make fraudulent transactions more extreme
    X[:n_fraud] *= 2
    
    # Create DataFrame
    columns = [f'V{i+1}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=columns)
    df['Class'] = y
    
    return df

def preprocess_data(df):
    """Preprocess the data and handle class imbalance."""
    print("Preprocessing data...")
    
    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    return X_train_balanced, X_test_scaled, y_train_balanced, y_test

def train_models(X_train, y_train):
    """Train multiple models and return them."""
    print("Training models...")
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42)
    }
    
    # Train each model
    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    return trained_models

def evaluate_models(models, X_test, y_test):
    """Evaluate models and print metrics."""
    print("\nModel Evaluation:")
    print("-" * 50)
    
    for name, model in models.items():
        print(f"\n{name} Results:")
        y_pred = model.predict(X_test)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

def plot_feature_importance(model, feature_names, model_name):
    """Plot feature importance for tree-based models."""
    if hasattr(model, 'feature_importances_'):
        plt.figure(figsize=(10, 6))
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.title(f'Feature Importance - {model_name}')
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.show()

def main():
    # Generate synthetic data
    df = generate_synthetic_data(n_samples=1000, fraud_ratio=0.1)
    
    # Print dataset information
    print("\nDataset Information:")
    print(f"Total number of transactions: {len(df)}")
    print(f"Number of fraudulent transactions: {df['Class'].sum()}")
    print(f"Number of legitimate transactions: {len(df) - df['Class'].sum()}")
    
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    # Train models
    models = train_models(X_train, y_train)
    
    # Evaluate models
    evaluate_models(models, X_test, y_test)
    
    # Plot feature importance for tree-based models
    feature_names = df.drop('Class', axis=1).columns
    for name in ['Random Forest', 'Decision Tree']:
        plot_feature_importance(models[name], feature_names, name)

if __name__ == "__main__":
    main() 