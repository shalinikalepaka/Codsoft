import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Join tokens back into string
    return ' '.join(tokens)

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def main():
    # Load the dataset
    # Note: You'll need to download the dataset from Kaggle and place it in the same directory
    try:
        df = pd.read_csv('spam.csv', encoding='latin-1')
        # Keep only the necessary columns
        df = df[['v1', 'v2']]
        df.columns = ['label', 'message']
    except FileNotFoundError:
        print("Please download the dataset from Kaggle and place it in the same directory as this script.")
        return

    # Preprocess the messages
    print("Preprocessing messages...")
    df['processed_message'] = df['message'].apply(preprocess_text)

    # Convert labels to binary
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_message'], df['label'], test_size=0.2, random_state=42
    )

    # Create TF-IDF vectors
    print("Creating TF-IDF vectors...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train and evaluate different models
    models = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'SVM': SVC(kernel='linear')
    }

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        
        print(f"\n{name} Results:")
        print(classification_report(y_test, y_pred))
        plot_confusion_matrix(y_test, y_pred, f'Confusion Matrix - {name}')

if __name__ == "__main__":
    main() 