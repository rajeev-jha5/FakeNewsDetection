# Fake News Detection using Machine Learning (TF-IDF + Logistic Regression)

## Project Overview

This project implements a **Fake News Detection** system using **Natural Language Processing (NLP)** and **Machine Learning**. The objective is to classify a news article as **Real** or **Fake** based on its title and article text.

The project uses the **Fake and Real News Dataset** from Kaggle and follows a complete machine learning pipeline including:

* Data Loading
* Exploratory Data Analysis (EDA)
* Text Preprocessing
* Feature Engineering using TF-IDF
* Logistic Regression Model Training
* Hyperparameter Tuning
* Model Evaluation
* Model Saving
* Prediction on New News Articles

The implementation follows the assignment requirement of a **70:10:20 Train : Validation : Test split** and is fully reproducible.

---

# Problem Statement

Given a news article consisting of a title and body text, build a machine learning model that classifies the article into one of the following categories:

* **Real News (Label = 1)**
* **Fake News (Label = 0)**

This is a **Supervised Binary Classification** problem.

---

# Dataset

Dataset Name:

**Fake and Real News Dataset**

Files:

* Fake.csv
* True.csv

Columns:

* title
* text
* subject
* date

After loading:

* Fake articles are assigned label **0**
* Real articles are assigned label **1**

---

# Project Structure

```text
FakeNewsDetection/

│
├── data/
│   ├── Fake.csv
│   ├── True.csv
│   ├── merged_news.csv
│   └── preprocessed_news.csv
│
├── models/
│   ├── logistic_regression.pkl
│   └── tfidf_vectorizer.pkl
│
├── outputs/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── metrics.txt
│   └── misclassified_articles.csv
│
├── notebooks/
│   └── FakeNewsDetection.ipynb
│
├── requirements.txt
├── README.md
└── LICENSE (optional)
```

---

# Technologies Used

* Python 3.11+
* Pandas
* NumPy
* Scikit-learn
* NLTK
* Matplotlib
* Joblib

---

# Machine Learning Workflow

1. Load Dataset
2. Merge Fake and Real News
3. Perform Exploratory Data Analysis
4. Clean and preprocess text
5. Split data into Train, Validation and Test sets
6. Convert text into TF-IDF vectors
7. Train Logistic Regression classifier
8. Tune hyperparameters using GridSearchCV
9. Evaluate the model
10. Save trained model and vectorizer
11. Predict on unseen news articles

---

# Data Preprocessing

The following preprocessing steps were applied:

* Convert text to lowercase
* Remove HTML tags
* Remove URLs
* Remove numbers
* Remove punctuation
* Remove extra whitespace
* Tokenization
* Stopword removal
* Lemmatization
* Combine title and article body

---

# Feature Engineering

TF-IDF Vectorizer parameters:

* max_features = 5000
* ngram_range = (1,2)
* min_df = 5
* max_df = 0.8

---

# Machine Learning Model

Algorithm:

**Logistic Regression**

Hyperparameters:

* solver = liblinear
* C = 1.0
* max_iter = 1000
* random_state = 42

---

# Train–Validation–Test Split

The dataset was divided using stratified sampling:

* Training Set = 70%
* Validation Set = 10%
* Test Set = 20%

Random seed:

```python
random_state = 42
```

---

# Evaluation Metrics

The model was evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix
* Classification Report

---

# How to Run the Project

## Step 1

Clone or download this project.

## Step 2

Install all required dependencies.

```bash
pip install -r requirements.txt
```

## Step 3

Place the following dataset files inside the **data** folder:

* Fake.csv
* True.csv

## Step 4

Open the Jupyter Notebook:

```
FakeNewsDetection.ipynb
```

Run all notebook cells in order.

The notebook performs:

* Data Loading
* EDA
* Text Preprocessing
* TF-IDF Vectorization
* Model Training
* Hyperparameter Tuning
* Model Evaluation
* Model Saving
* Prediction on New Data

---

# Saved Files

After execution, the following files are generated.

## Models

* models/logistic_regression.pkl
* models/tfidf_vectorizer.pkl

## Outputs

* confusion_matrix.png
* roc_curve.png
* metrics.txt
* misclassified_articles.csv

---

# Sample Prediction

Input:

```
Scientists discovered a new vaccine effective against multiple virus strains.
```

Output:

```
Prediction : Real News

Probability : 0.97
```

---

# Reproducibility

This project is fully reproducible because:

* Fixed random seed (42)
* Fixed Train–Validation–Test split
* Same preprocessing pipeline
* Fixed TF-IDF parameters
* Fixed Logistic Regression hyperparameters
* Saved TF-IDF vectorizer
* Saved trained model
* Documented Python libraries

---

# Limitations

The current model:

* Does not understand semantic meaning.
* Does not understand sarcasm.
* Does not capture long-range context.
* Relies on TF-IDF features.
* May misclassify satire or highly ambiguous articles.

---

# Future Improvements

Possible improvements include:

* Linear SVM
* Random Forest
* XGBoost
* Character n-grams
* Word2Vec
* GloVe
* FastText
* BERT
* RoBERTa
* DistilBERT

---

# Results

The exact results depend on the train/validation/test split and software versions.

Typical performance on this dataset is:

| Metric    | Expected Range |
| --------- | -------------- |
| Accuracy  | 97%–99%        |
| Precision | 97%–99%        |
| Recall    | 97%–99%        |
| F1 Score  | 97%–99%        |
| ROC-AUC   | 0.99–1.00      |

---

# Conclusion

This project demonstrates an end-to-end machine learning solution for fake news detection using TF-IDF and Logistic Regression. It follows a standard NLP pipeline, produces highly competitive performance on the Fake and Real News Dataset, and emphasizes reproducibility, interpretability, and sound machine learning practices.

---

# Author

**Rajeev Kumar**

Machine Learning | Data Science | Natural Language Processing
