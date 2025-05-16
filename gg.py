import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import streamlit as st
import joblib
import os

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Load and combine datasets
def load_data():
    data1 = pd.read_excel('NLP Project A dataset.xlsx', sheet_name='data1')
    data2 = pd.read_excel('NLP Project A dataset.xlsx', sheet_name='data2')
    data3 = pd.read_excel('NLP Project A dataset.xlsx', sheet_name='data3')
    df = pd.concat([data1, data2, data3], ignore_index=True)
    df.dropna(inplace=True)
    df.drop_duplicates(subset=['utterance'], inplace=True)
    return df

# Prepare data for models
def prepare_data(df):
    df['processed'] = df['utterance'].apply(preprocess_text)
    tfidf = TfidfVectorizer(max_features=5000)
    X_tfidf = tfidf.fit_transform(df['processed']).toarray()
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(df['processed'])
    X_seq = tokenizer.texts_to_sequences(df['processed'])
    X_pad = pad_sequences(X_seq, maxlen=100)
    le = LabelEncoder()
    y = le.fit_transform(df['category'])
    return X_tfidf, X_pad, y, tfidf, tokenizer, le

# Train or load XGBoost
def train_xgboost(X, y):
    model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X, y)
    joblib.dump(model, "xgboost_model.pkl")
    return model

# Train or load LSTM
def train_lstm(X, y, num_classes):
    model = Sequential([
        LSTM(128, input_shape=(X.shape[1], 1)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X.reshape((X.shape[0], X.shape[1], 1)), y, epochs=10, batch_size=32, validation_split=0.2)
    model.save("lstm_model.h5")
    return model

# Ensemble prediction
def ensemble_predict(text, tfidf, tokenizer, max_len, xgb, lstm, le):
    proc = preprocess_text(text)
    x_tfidf = tfidf.transform([proc]).toarray()
    x_seq = tokenizer.texts_to_sequences([proc])
    x_pad = pad_sequences(x_seq, maxlen=max_len).reshape((1, max_len, 1))
    pred_xgb = xgb.predict_proba(x_tfidf)[0]
    pred_lstm = lstm.predict(x_pad, verbose=0)[0]
    avg = (pred_xgb + pred_lstm) / 2
    return le.inverse_transform([np.argmax(avg)])[0]

# Smart reply logic
def generate_reply(category):
    responses = {
        "refund": "I understand you're requesting a refund. I've assigned this to our refund department.",
        "payment": "Your query is related to payment. Please wait while we check your status.",
        "feedback": "Thank you for your feedback! We’ll forward it to our improvement team.",
        "delivery": "Delivery status is being verified. Please hold on.",
        "shipping address": "We’ll help you update your shipping address shortly.",
        "invoice": "You can download your invoice from your account page.",
        "account": "We’ll help you with your account-related issue shortly.",
        "newsletter": "You're being unsubscribed from our newsletter. You'll receive a confirmation soon.",
        "contact": "We’ve noted your contact request. A support agent will reach out."
    }
    return responses.get(category.lower(), "Thank you! We're processing your request.")

# Streamlit chatbot UI
def chatbot():
    st.title("Smart Customer Support Chatbot")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.chat_input("Ask a question about your order...")
    if user_input:
        st.chat_message("user").markdown(user_input)
        category = ensemble_predict(user_input, tfidf, tokenizer, 100, xgb_model, lstm_model, le)
        response = generate_reply(category)
        st.chat_message("assistant").markdown(f"**Category:** {category}\n\n{response}")
        st.session_state.chat_history.append((user_input, category, response))

# Load or train models
if os.path.exists("xgboost_model.pkl") and os.path.exists("lstm_model.h5"):
    xgb_model = joblib.load("xgboost_model.pkl")
    lstm_model = load_model("lstm_model.h5")
    df = load_data()
    X_tfidf, X_pad, y, tfidf, tokenizer, le = prepare_data(df)
else:
    df = load_data()
    X_tfidf, X_pad, y, tfidf, tokenizer, le = prepare_data(df)
    xgb_model = train_xgboost(X_tfidf, y)
    lstm_model = train_lstm(X_pad, y, len(le.classes_))

# Run the chatbot
chatbot()
