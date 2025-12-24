import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression

# --- STEP 1: DATA PREPARATION ---
# Asli project mein aap pd.read_csv('data.csv') use karenge
def load_data():
    data = {
        'product_id': [101, 102, 103, 104, 105],
        'product_name': ['Apple iPhone 15', 'Samsung S23 Ultra', 'Sony WH-1000XM5', 'Bose QuietComfort', 'Apple MacBook Pro'],
        'category': ['Electronics', 'Electronics', 'Audio', 'Audio', 'Computers'],
        'description': ['high end smartphone ios camera', 'android flagship phone stylus zoom', 'noise cancelling headphones wireless', 'premium comfort audio headphones', 'powerful laptop m3 chip retina display'],
        'review_text': ['amazing phone great camera', 'best android screen quality', 'noise cancellation is weak', 'very comfortable for long hours', 'insane performance and battery'],
        'sentiment_label': [1, 1, 0, 1, 1] # 1 = Positive, 0 = Negative
    }
    return pd.DataFrame(data)

df = load_data()

# --- STEP 2: ML MODELS SETUP ---

# A. Sentiment Model (NLP)
tfidf_sent = TfidfVectorizer(stop_words='english')
X_sent = tfidf_sent.fit_transform(df['review_text'])
y_sent = df['sentiment_label']
model_sent = LogisticRegression().fit(X_sent, y_sent)

# B. Recommendation Engine (Cosine Similarity)
tfidf_rec = TfidfVectorizer(stop_words='english')
matrix_rec = tfidf_rec.fit_transform(df['description'])
sim_score = cosine_similarity(matrix_rec)

# --- STEP 3: STREAMLIT UI ---

st.set_page_config(page_title="AI E-Commerce Suite", layout="wide")
st.title("🚀 Smart E-Commerce Sentiment & Recommendation System")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.header("🔍 Sentiment Analysis")
    user_input = st.text_area("Customer ka review yahan likhein:", placeholder="Example: The product quality is bad...")
    
    if st.button("Analyze Sentiment"):
        if user_input:
            vec = tfidf_sent.transform([user_input])
            pred = model_sent.predict(vec)[0]
            label = "✅ POSITIVE" if pred == 1 else "❌ NEGATIVE"
            color = "green" if pred == 1 else "red"
            st.markdown(f"### Result: :{color}[{label}]")
        else:
            st.warning("Kuch toh likhiye!")

with col2:
    st.header("🎁 Product Recommendations")
    selected_prod = st.selectbox("Product select karein:", df['product_name'])
    
    if st.button("Recommend Similar Items"):
        idx = df[df['product_name'] == selected_prod].index[0]
        distances = sorted(list(enumerate(sim_score[idx])), reverse=True, key=lambda x: x[1])
        
        st.write("Aapko ye bhi pasand aa sakte hain:")
        for i in distances[1:3]: # Top 2 recommendations
            st.success(df.iloc[i[0]]['product_name'])

st.sidebar.info("Ye system Machine Learning (NLP) aur Cosine Similarity par chalta hai.")