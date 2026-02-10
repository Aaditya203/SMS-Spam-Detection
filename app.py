import streamlit as st
import pickle
import nltk
import string

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('punkt_tab')
st.set_page_config(page_title="SMS Spam Classifier", page_icon="📩")
# download required nltk data (needed for deployment)
nltk.download('punkt')
nltk.download('stopwords')

# initialize stemmer and stopwords
ps = PorterStemmer()
stopwords_list = set(stopwords.words('english'))
exclude = string.punctuation

# ---------- PREPROCESS FUNCTION ----------
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', exclude))
    words = nltk.word_tokenize(text)
    
    processed_words = []
    for word in words:
        if word.isalnum() and word not in stopwords_list:
            processed_words.append(ps.stem(word))
    
    return " ".join(processed_words)

# ---------- LOAD MODEL & VECTORIZER ----------
@st.cache_resource
def load_model():
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return vectorizer, model

vectorizer, model = load_model()

# ---------- STREAMLIT UI ----------

st.title("📩 SMS Spam Classifier")
st.write("Enter an SMS message to check whether it is **Spam** or **Not Spam**.")

user_input = st.text_area("Enter SMS text here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # preprocess
        processed_text = preprocess(user_input)
        
        # vectorize
        vector_input = vectorizer.transform([processed_text])
        
        # predict
        prediction = model.predict(vector_input)[0]
        
        if prediction == 1:
            st.error("🚨 This message is **SPAM**")
        else:
            st.success("✅ This message is **NOT SPAM**")
