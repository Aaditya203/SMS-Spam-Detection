# SMS Spam Detection using Machine Learning

This project is an end-to-end SMS Spam Classification system built using Natural Language Processing and Machine Learning.  
It is deployed as a web application using Streamlit.

---

## 🔹 Features
- Text preprocessing (lowercasing, punctuation removal, stopwords removal, stemming)
- Vectorization using trained vectorizer
- Machine learning model for spam detection
- Interactive Streamlit web interface
- Deployed on Streamlit

---

## 🔹 Tech Stack
- Python
- NLTK
- Scikit-learn
- Streamlit


---

## 🔹 How it works
1. User enters an SMS message
2. Text is preprocessed using NLP techniques
3. Text is converted into numerical features
4. ML model predicts whether the SMS is Spam or Not Spam

---

## 🔹 Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
