# 🧠 Social-Media-Sentiment-Analysis-using-DM-and-ML
This project performs sentiment analysis on tweets using multiple machine learning models. It uses the **Sentiment140** dataset and compares different models with both **CountVectorizer** and **TfidfVectorizer** techniques. The best-performing model is then saved for deployment.

## 📌 Features

* Downloads and processes the **Sentiment140** dataset from Kaggle.
* Performs data cleaning and preprocessing including stemming and noise removal.
* Visualizes sentiment distributions and word clouds.
* Trains and evaluates multiple ML models:

  * Bernoulli Naive Bayes
  * Multinomial Naive Bayes
  * Logistic Regression
  * Linear SVC
* Saves the best-performing model (`Logistic Regression with TfidfVectorizer`) using `pickle`.

---

## 🗂️ Dataset

* Source: [Sentiment140 on Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)
* Contains 1.6M tweets labeled as:

  * `0` → Negative
  * `4` → Positive (converted to `1` in the script)

---

## ⚙️ Tech Stack

* Python
* Pandas, NumPy
* NLTK for text preprocessing
* Scikit-learn for ML models
* Seaborn, Matplotlib, WordCloud for visualization
* Pickle for model serialization

---

## 🚀 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/twitter-sentiment-analysis.git
   cd twitter-sentiment-analysis
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Place your `kaggle.json` file in the root directory to access the dataset.

4. Run the Python script:

   ```bash
   python sentiment_analysis.py
   ```

---

## 🧪 Model Evaluation Summary

| ID      | Vectorizer      | Model              | Accuracy | f1-score (0) | f1-score (1) | Training Time |
| ------- | --------------- | ------------------ | -------- | ------------ | ------------ | ------------- |
| Model-1 | CountVectorizer | BernoulliNB        | 0.79     | 0.79         | 0.79         | 0.93 sec      |
| Model-2 | CountVectorizer | MultinomialNB      | 0.80     | 0.80         | 0.79         | 0.87 sec      |
| Model-3 | CountVectorizer | LogisticRegression | 0.80     | 0.80         | 0.81         | 531.09 sec    |
| Model-4 | CountVectorizer | LinearSVC          | 0.78     | 0.78         | 0.78         | 680.62 sec    |
| Model-5 | TfidfVectorizer | BernoulliNB        | 0.79     | 0.79         | 0.79         | 1.07 sec      |
| Model-6 | TfidfVectorizer | MultinomialNB      | 0.80     | 0.80         | 0.79         | 0.71 sec      |
| Model-7 | TfidfVectorizer | LogisticRegression | **0.82** | **0.82**     | **0.82**     | 22.81 sec     |
| Model-8 | TfidfVectorizer | LinearSVC          | 0.81     | 0.81         | 0.81         | 52.50 sec     |

---

## 📦 Output

* `vectorizer.pkl` – TfidfVectorizer model
* `model.pkl` – Trained Logistic Regression model

---

## 📝 License

This project is accepted in the ICRTICC Conference.
