# Movie Genre Classification

This project implements a simple machine learning model to predict movie genres based on plot summaries. It uses TF-IDF vectorization and a Naive Bayes classifier to make predictions.

## Features

- Text preprocessing
- TF-IDF vectorization
- Naive Bayes classification
- Model evaluation metrics
- Example predictions

## Requirements

- Python 3.7+
- Required packages are listed in `requirements.txt`

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. The dataset is provided in `movies.csv`
2. Run the classifier:
```bash
python movie_genre_classifier.py
```

The script will:
- Load and preprocess the data
- Split it into training and test sets
- Train the model
- Show evaluation metrics
- Make an example prediction

## Dataset

The included dataset contains 10 movies with their plots and genres. The genres include:
- Action
- Romance
- Crime
- Sci-Fi
- Drama
- Thriller

## How it Works

1. The text data is preprocessed (converted to lowercase)
2. TF-IDF vectorization converts the text into numerical features
3. A Naive Bayes classifier is trained on these features
4. The model can then predict genres for new movie plots

## Example

You can test the model with your own movie plots by modifying the `new_plot` variable in the `main()` function of `movie_genre_classifier.py`. 