import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
def load_data():
    return pd.read_csv('movies.csv')

# Preprocess the data
def preprocess_data(df):
    # Convert text to lowercase
    df['plot'] = df['plot'].str.lower()
    return df

# Train the models
def train_models(X_train, y_train):
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    # Initialize models
    models = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'SVM': SVC(kernel='linear')
    }
    
    # Train each model
    trained_models = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_tfidf, y_train)
        trained_models[name] = model
    
    return vectorizer, trained_models

# Evaluate the models
def evaluate_models(vectorizer, models, X_test, y_test):
    X_test_tfidf = vectorizer.transform(X_test)
    results = {}
    
    for name, model in models.items():
        print(f"\n{'-'*50}")
        print(f"Results for {name}:")
        print(f"{'-'*50}")
        
        y_pred = model.predict(X_test_tfidf)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nAccuracy:", accuracy_score(y_test, y_pred))
        
        results[name] = {
            'predictions': y_pred,
            'accuracy': accuracy_score(y_test, y_pred)
        }
    
    return results

# Predict genre for new movie plot
def predict_genre(vectorizer, models, plot):
    plot_tfidf = vectorizer.transform([plot.lower()])
    predictions = {}
    
    for name, model in models.items():
        prediction = model.predict(plot_tfidf)
        predictions[name] = prediction[0]
    
    return predictions

def plot_accuracy_comparison(results):
    # Create a figure with a single subplot
    plt.figure(figsize=(10, 6))
    
    # Plot bar chart of accuracies
    models = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in models]
    
    bars = plt.bar(models, accuracies)
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    
    # Add accuracy values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()

def plot_confusion_matrices(results, y_test):
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(15, 5))
    
    for idx, (model_name, result) in enumerate(results.items()):
        cm = confusion_matrix(y_test, result['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=sorted(set(y_test)),
                   yticklabels=sorted(set(y_test)),
                   ax=axes[idx])
        axes[idx].set_title(f'{model_name} Confusion Matrix')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    plt.close()

def plot_feature_importance(vectorizer, models, X_train, y_train):
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    y_classes = sorted(set(y_train))
    n_classes = len(y_classes)
    if n_classes < 2:
        print('Not enough classes for feature importance plot.')
        return
    # Create a figure for each model
    for model_name, model in models.items():
        if hasattr(model, 'coef_') or model_name == 'Naive Bayes':
            # Get coefficients and model classes
            if model_name == 'Naive Bayes':
                coef = model.feature_log_prob_
                model_classes = model.classes_
            else:
                coef = model.coef_
                model_classes = model.classes_
            coef = np.array(coef)  # Ensure dense array
            # Only plot if number of classes matches and coef is 2D
            if coef.ndim != 2 or coef.shape[0] != len(model_classes) or coef.shape[0] != n_classes:
                print(f'Skipping {model_name} feature importance: class/coef mismatch or not multiclass.')
                continue
            plt.figure(figsize=(12, 6))
            for i, genre in enumerate(y_classes):
                if genre not in model_classes:
                    continue
                class_idx = list(model_classes).index(genre)
                top_features = np.argsort(coef[class_idx])[-10:]
                plt.subplot(1, n_classes, i+1)
                plt.barh(range(10), coef[class_idx][top_features])
                plt.yticks(range(10), [feature_names[j] for j in top_features])
                plt.title(f'Top Features for {genre}')
            plt.tight_layout()
            plt.savefig(f'{model_name}_feature_importance.png')
            plt.close()

def main():
    # Load and preprocess data
    df = load_data()
    df = preprocess_data(df)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        df['plot'], df['genre'], test_size=0.2, random_state=42
    )
    
    # Train the models
    vectorizer, models = train_models(X_train, y_train)
    
    # Evaluate the models
    results = evaluate_models(vectorizer, models, X_test, y_test)
    
    # Create visualizations
    plot_accuracy_comparison(results)
    plot_confusion_matrices(results, y_test)
    plot_feature_importance(vectorizer, models, X_train, y_train)
    
    # Example prediction
    new_plot = "A young wizard discovers his magical abilities and battles against an evil dark lord."
    predictions = predict_genre(vectorizer, models, new_plot)
    
    print(f"\nExample prediction for new plot:")
    print(f"Plot: {new_plot}")
    print("\nPredictions from different models:")
    for model_name, prediction in predictions.items():
        print(f"{model_name}: {prediction}")

if __name__ == "__main__":
    main() 