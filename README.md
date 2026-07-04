# Fake News Detection using TF-IDF and Logistic Regression

## Project Overview

This project implements a Fake News Detection system using Natural Language Processing (NLP) and Machine Learning. The objective is to classify a news article as **Real** or **Fake** using the Fake and Real News Dataset.

The project follows a complete machine learning pipeline including data preprocessing, feature engineering using TF-IDF, model training with Logistic Regression, hyperparameter tuning, and evaluation.

---

## Dataset

Dataset Used: **Fake and Real News Dataset**

Files:

- `Fake.csv`
- `True.csv`

The dataset is placed inside the **data/** folder.

---

## Project Structure

```
FakeNewsDetection/

│── data/
│   ├── Fake.csv
│   ├── True.csv
│
│── models/
│   ├── logistic_regression.pkl
│   ├── tfidf_vectorizer.pkl
│
│── outputs/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── metrics.txt
│   ├── misclassified_articles.csv
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

## Machine Learning Pipeline

1. Load Dataset
2. Exploratory Data Analysis (EDA)
3. Text Preprocessing
4. Train–Validation–Test Split (70:10:20)
5. TF-IDF Feature Engineering
6. Logistic Regression Training
7. Hyperparameter Tuning
8. Model Evaluation
9. Save Model
10. Predict New News Articles

---

## Text Preprocessing

The following preprocessing steps were applied:

- Convert text to lowercase
- Remove HTML tags
- Remove URLs
- Remove numbers
- Remove punctuation
- Remove extra spaces
- Tokenization
- Stopword Removal
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

Parameters:

- solver = liblinear
- C = 1.0
- max_iter = 1000
- random_state = 42

---

## Data Split

- Training Set : 70%
- Validation Set : 10%
- Test Set : 20%

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

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Place Dataset

Copy the following files inside the **data/** folder:

- Fake.csv
- True.csv

### 3. Run the Notebooks

Run the notebooks in the following order:

1. **EDA.ipynb**
2. **FakeNewsDetection.ipynb**

The notebook performs:

- Data Loading
- Data Preprocessing
- TF-IDF Feature Engineering
- Model Training
- Hyperparameter Tuning
- Model Evaluation
- Model Saving
- Prediction

---

## Output Files

After successful execution, the following files will be generated.

### models/

- logistic_regression.pkl
- tfidf_vectorizer.pkl

### outputs/

- confusion_matrix.png
- roc_curve.png
- metrics.txt
- misclassified_articles.csv

---

## Reproducibility

The project is fully reproducible because:

- Fixed random seed (`random_state = 42`)
- Same preprocessing pipeline
- Fixed train-validation-test split
- Same TF-IDF parameters
- Fixed Logistic Regression hyperparameters
- Saved trained model
- Saved TF-IDF vectorizer

---

## Future Improvements

- Linear SVM
- Random Forest
- XGBoost
- BERT
- RoBERTa
- DistilBERT

---

## Author

**Rajeev Kumar**
