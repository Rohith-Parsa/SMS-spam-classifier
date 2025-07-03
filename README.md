# SMS Spam Classifier

A machine learning project that classifies SMS messages as **spam** or **ham** using Natural Language Processing (NLP) and various classifiers like Naive Bayes, Logistic Regression, SVM, and Random Forest.

---

## Features

- Exploratory Data Analysis (EDA) and visualization
- WordClouds for spam and ham messages
- TF-IDF vectorization for text features
- Comparison of different ML models
- Hyperparameter tuning with GridSearchCV
- Achieved ~95% accuracy with SVM

---

##  Dataset

- Dataset: [Spam SMS Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- Format: CSV (`spam.csv`)
- Labels:
  - `ham` → Not spam
  - `spam` → Unwanted message

---

##  EDA & Visualization

- Spam vs Ham message count
- Message length distribution
- WordCloud of common words in spam and ham

---

##  Models Used

| Model                  | Accuracy |
|------------------------|----------|
| Multinomial Naive Bayes| ~96%     |
| Logistic Regression    | ~96%     |
| Support Vector Machine | ~98%   |
| Random Forest          | ~97%     |

> Final model selected using `GridSearchCV` with SVM pipeline

---

##  Tech Stack

- Python
- Pandas, NumPy
- Seaborn, Matplotlib
- Scikit-learn
- WordCloud
- Jupyter Notebook

---

##  How to Use

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/sms-spam-classifier.git
   cd sms-spam-classifier
