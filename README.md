# Fake News Detection using TF-IDF and Logistic Regression

## Project Overview

This project implements a Fake News Detection system using Natural Language Processing (NLP) and Machine Learning. The objective is to classify a news article as **Real** or **Fake** using the Fake and Real News Dataset.

The project follows a complete machine learning pipeline including Exploratory Data Analysis (EDA), text preprocessing, TF-IDF feature extraction, Logistic Regression model training, evaluation, and prediction.

---

## Dataset

**Dataset:** Fake and Real News Dataset

Files used:

- Fake.csv
- True.csv

The dataset contains news articles with the following columns:

- title
- text
- subject
- date

Labels assigned:

- Fake News → **0**
- Real News → **1**

---

## Project Structure

```
FakeNewsDetection/

│── data/
│   ├── Fake.csv
│   ├── True.csv
│   ├── merged_news.csv
│   └── preprocessed_news.csv
│
│── models/
│   ├── logistic_regression.pkl
│   └── tfidf_vectorizer.pkl
│
│── outputs/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── metrics.txt
│   └── misclassified_articles.csv
│
│── EDA.ipynb
│── FakeNewsDetection.ipynb
│── requirements.txt
│── README.md
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Matplotlib
- Joblib

---

## Methodology

The project follows these steps:

1. Load the dataset
2. Perform Exploratory Data Analysis (EDA)
3. Clean and preprocess the text
4. Split the dataset into Train (70%), Validation (10%), and Test (20%)
5. Convert text into TF-IDF vectors
6. Train a Logistic Regression classifier
7. Evaluate the model
8. Save the trained model and TF-IDF vectorizer
9. Predict new news articles

---

## Text Preprocessing

The following preprocessing steps were applied:

- Convert text to lowercase
- Remove HTML tags
- Remove URLs
- Remove numbers
- Remove punctuation
- Remove extra whitespaces
- Tokenization
- Stopword removal
- Lemmatization
- Combine title and article text

---

## Feature Engineering

TF-IDF Parameters:

- max_features = 5000
- ngram_range = (1,2)
- min_df = 5
- max_df = 0.8

---

## Machine Learning Model

**Algorithm:** Logistic Regression

Model Parameters:

- solver = liblinear
- C = 1.0
- max_iter = 1000
- random_state = 42

---

## Evaluation Metrics

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix
- Classification Report

---

## How to Run

### 1. Install the required libraries

```bash
pip install -r requirements.txt
```

### 2. Place the dataset files inside the `data` folder

- Fake.csv
- True.csv

### 3. Run the notebooks in the following order

1. `EDA.ipynb`
2. `FakeNewsDetection.ipynb`

The notebooks perform:

- Data Loading
- Data Preprocessing
- TF-IDF Feature Engineering
- Model Training
- Model Evaluation
- Model Saving
- Prediction on New News

---

## Output Files

### Models

- models/logistic_regression.pkl
- models/tfidf_vectorizer.pkl

### Outputs

- outputs/confusion_matrix.png
- outputs/roc_curve.png
- outputs/misclassified_articles.csv

---

## Reproducibility

This project is reproducible because:

- Fixed random seed (`random_state = 42`)
- Fixed Train–Validation–Test split (70:10:20)
- Consistent preprocessing pipeline
- Fixed TF-IDF parameters
- Fixed Logistic Regression parameters
- Saved trained model
- Saved TF-IDF vectorizer

---

## Future Improvements

Possible improvements include:

- Support Vector Machine (SVM)
- XGBoost
- BERT
- RoBERTa
- DistilBERT

---

## Author

**Rajeev Kumar**
