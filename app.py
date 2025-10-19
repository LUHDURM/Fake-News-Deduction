# app.py

import streamlit as st
import joblib

# Load the saved model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Streamlit page setup
st.set_page_config(page_title="Fake News Detector", page_icon="", layout="centered")

st.title("Fake News Detection System")
st.write("This app uses **Machine Learning (Logistic Regression)** to classify whether a news article is *Real* or *Fake*.")

# Text input area
user_input = st.text_area("Enter the news article text below:", height=200)

if st.button("Analyze News"):
    if user_input.strip() == "":
        st.warning("Please enter some news content before analyzing.")
    else:
        # Vectorize the input text
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]

        # Display result
        if prediction == 0:
            st.success("The news is **REAL**.")
        else:
            st.error("The news is **FAKE**.")

# Sidebar info
st.sidebar.title("About the Model")
st.sidebar.markdown("""
**Algorithm Used:** Logistic Regression  
**Feature Extraction:** TF-IDF Vectorizer  

**Labels:**  
- 0 → Real News  
- 1 → Fake News  

**Why Logistic Regression?**  
It’s simple, efficient, and performs well for binary text classification problems.
""")

# Footer
st.markdown("---")
st.caption("Developed by **Mrudhul NR** | Fake News Detection Project")
