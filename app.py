import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required nltk data
nltk.download('punkt')
nltk.download('stopwords')

# Load model and vectorizer
with open("fake_news_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer_fake_news.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Stopwords set
stop_word = set(stopwords.words('english'))

# remove stopwords
def remove_stopwords(txt):
    txt = txt.lower()
    txt = re.sub(r'[^a-zA-Z ]', ' ', txt)
    tokens = word_tokenize(txt)

    text = []
    for word in tokens:
        if word not in stop_word:
            text.append(word)

    return " ".join(text)

# UI
st.set_page_config(page_title="Fake News Detector", page_icon="📰")

st.title("📰 Fake News Detection App")
st.markdown("### Enter News Title + Content Below")

user_input = st.text_area("Enter News Here")

if st.button("Predict"):

    if user_input.strip() == "":
        st.warning("⚠ Please enter some text.")
    else:
        # SAME preprocessing as training
        cleaned_text = remove_stopwords(user_input)

        vectorized_text = vectorizer.transform([cleaned_text])
        prediction = model.predict(vectorized_text)[0]
        probability = model.predict_proba(vectorized_text)[0][1]

        if prediction == 1:
            st.error("🚨 This News is FAKE")
        else:
            st.success("✅ This News is REAL")

        st.write("Confidence:", round(probability * 100, 2), "%")



st.markdown("---")
st.markdown("## 👨‍💻 About Me")

st.write("""
Hello! I am **Alok Sahu**.  
This Fake News Detection App is built using:

- TF-IDF Vectorizer  
- Logistic Regression  
- Streamlit  

The model achieves around **95% accuracy** on the test dataset. """)

