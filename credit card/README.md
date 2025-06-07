# Credit Card Fraud Detection

This project implements a machine learning model to detect fraudulent credit card transactions. The model uses various algorithms to classify transactions as either fraudulent or legitimate.

## Dataset
The project uses a credit card fraud dataset from Kaggle. The dataset contains anonymized credit card transactions, where each transaction is labeled as either fraudulent (1) or legitimate (0).

## Features
- Data preprocessing and feature engineering
- Multiple machine learning models (Logistic Regression, Random Forest)
- Model evaluation metrics (Precision, Recall, F1-score)
- Handling class imbalance using SMOTE
- Visualization of results

## Setup
1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Download the dataset from Kaggle and place it in the `data` directory.

3. Run the main script:
```bash
python credit_card_fraud_detection.py
```

## Project Structure
- `credit_card_fraud_detection.py`: Main script containing the model implementation
- `requirements.txt`: Project dependencies
- `data/`: Directory containing the dataset 