# SMS Spam Detection

This project implements a machine learning model to classify SMS messages as spam or legitimate (ham) using various classification algorithms.

## Features

- Text preprocessing using NLTK
- TF-IDF vectorization
- Multiple classifier implementations:
  - Naive Bayes
  - Logistic Regression
  - Support Vector Machine (SVM)
- Model evaluation with confusion matrices and classification reports

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Download the dataset:
   - Go to [Kaggle SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
   - Download the dataset
   - Place the `spam.csv` file in the same directory as the script

## Usage

Run the script:
```bash
python spam_detection.py
```

The script will:
1. Load and preprocess the data
2. Train multiple models
3. Display performance metrics for each model
4. Show confusion matrices

## Model Performance

The script evaluates three different models and provides:
- Classification report (precision, recall, F1-score)
- Confusion matrix visualization
- Overall accuracy metrics

## Note

The first time you run the script, it will download required NLTK data automatically. 