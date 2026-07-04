# Reproducibility Checklist

## Project Information

**Project Title:** Fake News Detection using TF-IDF and Logistic Regression

**Author:** Rajeev Kumar

---

## Dataset

- [x] Original dataset used without modification
- [x] Dataset source documented
- [x] Labels clearly defined
- [x] Dataset stored in the `data/` folder

---

## Environment

- [x] Python version documented
- [x] All required libraries listed in `requirements.txt`
- [x] Fixed library versions used

---

## Data Preprocessing

- [x] Combined title and article text
- [x] Converted text to lowercase
- [x] Removed HTML tags
- [x] Removed URLs
- [x] Removed numbers
- [x] Removed punctuation
- [x] Removed extra whitespaces
- [x] Tokenization performed
- [x] Stopwords removed
- [x] Lemmatization applied

---

## Data Splitting

- [x] Train : Validation : Test = 70 : 10 : 20
- [x] Stratified sampling used
- [x] Random seed fixed (`random_state = 42`)

---

## Feature Engineering

- [x] TF-IDF Vectorizer used
- [x] max_features = 5000
- [x] ngram_range = (1,2)
- [x] min_df = 5
- [x] max_df = 0.8

---

## Model

- [x] Logistic Regression used
- [x] solver = liblinear
- [x] C = 1.0
- [x] max_iter = 1000
- [x] random_state = 42

---

## Evaluation

- [x] Accuracy reported
- [x] Precision reported
- [x] Recall reported
- [x] F1-score reported
- [x] ROC-AUC reported
- [x] Confusion Matrix generated
- [x] Classification Report generated

---

## Saved Files

- [x] Trained model saved (`logistic_regression.pkl`)
- [x] TF-IDF vectorizer saved (`tfidf_vectorizer.pkl`)
- [x] Confusion Matrix saved
- [x] ROC Curve saved
- [x] Misclassified articles saved

---

## Documentation

- [x] README.md included
- [x] requirements.txt included
- [x] Project folder structure documented
- [x] Steps to run the project documented

---

## Reproducibility Statement

This project is fully reproducible. Running the notebooks with the provided dataset and the dependencies listed in `requirements.txt` will produce the same preprocessing pipeline, train-validation-test split, trained model, and evaluation results.
