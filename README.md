# 📰 Fake News Detection Web Application

A Machine Learning based web application that classifies news articles as **Fake** or **Real** using Natural Language Processing (NLP).

The model is built using **TF-IDF Vectorization + Logistic Regression** and deployed with **Streamlit**.

---

## 🚀 Project Overview

In the era of digital media, fake news spreads rapidly across online platforms.  
This project aims to automatically detect whether a news article is **Fake** or **Real** based on its textual content.

The model is trained on a labeled Kaggle dataset and achieves approximately **95% accuracy** on unseen test data.

---

## 📂 Dataset Information

📌 **Dataset Name:** Fake News Classification  
📌 **Source:** Kaggle  
🔗 https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification  

### Dataset Description

- Contains thousands of labeled news articles  
- Binary classification problem  
- Balanced dataset (Fake and Real nearly equal)

### Columns Used

- `title`
- `text`
- `label`

### Target Labels

- `0 → Real News`
- `1 → Fake News`


---

## 🧠 Machine Learning Pipeline

### 1️⃣ Text Preprocessing

- Lowercasing
- Removing special characters
- Stopwords removal
- Tokenization

### 2️⃣ Feature Extraction

- TF-IDF Vectorizer
- N-gram range: (1,2)
- Optimized max_features

### 3️⃣ Model Training

- Logistic Regression
- Regularization applied
- Train/Test Split (80/20)

---

## 📊 Model Performance

### ✅ Accuracy
**94.96%**

### 📈 Classification Report

| Class | Precision | Recall | F1-Score |
|-------|----------|--------|----------|
| Real  | 0.96 | 0.94 | 0.95 |
| Fake  | 0.94 | 0.96 | 0.95 |

### 📌 Confusion Matrix

|                | Predicted Real | Predicted Fake |
|---------------|----------------|----------------|
| **Actual Real** | 6604 | 477 |
| **Actual Fake** | 276  | 6951 |

### 🔎 Key Observations

- High recall (96%) for Fake news
- Low False Negative rate
- Balanced performance
- Minimal overfitting

---

## 🌐 Web Application (Streamlit)

The application allows users to:

- Enter news title and content
- Get instant prediction (Fake / Real)
- View confidence score
- Use a clean and interactive interface

---

## 🛠 Tech Stack

- Python
- Scikit-learn
- Numpy
- NLTK
- Streamlit
- Kaggle Dataset

---

## 📁 Project Structure


fake-news-detection/
│
├── app.py
├── fake_news_model.pkl
├── tfidf_vectorizer_fake_news.pkl
├── requirements.txt
└── README.md



---

## ▶️ How to Run Locally

### 1️⃣ Clone the Repository

### 2️⃣ Navigate to Project Folder

### 3️⃣ Install Dependencies

### 4️⃣ Run the Streamlit App



---

## 📌 Future Improvements

- Deploy on Streamlit Cloud
- Add Deep Learning model (LSTM / BERT)
- Add model comparison dashboard
- Improve UI/UX
- Create REST API version

---

## 👨‍💻 Author

**Alok Sahu**  
Machine Learning & Deep Learning Enthusiast 🚀  

Passionate about NLP, AI, and Data Science.

---

## ⭐ Support

If you found this project helpful, please consider giving it a ⭐ on GitHub.