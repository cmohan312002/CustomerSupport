import pandas as pd
import numpy as np
import streamlit as st
import joblib
import os
import json
import time
import plotly.express as px
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import re
from textblob import TextBlob

# Configure Streamlit page
st.set_page_config(
    page_title="Smart Customer Support AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download NLTK resources with error handling
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        return True
    except Exception as e:
        st.error(f"Failed to download NLTK resources: {e}")
        return False

download_nltk_data()

class CustomerSupportChatbot:
    def __init__(self):
        self.initialize_session_state()
        self.load_models_and_preprocessors()
        
    def initialize_session_state(self):
        """Initialize session state variables"""
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "conversation_stats" not in st.session_state:
            st.session_state.conversation_stats = {
                "total_queries": 0,
                "categories_count": {},
                "sentiment_scores": [],
                "response_times": [],
                "satisfaction_ratings": []
            }
        if "user_profile" not in st.session_state:
            st.session_state.user_profile = {
                "name": "",
                "email": "",
                "customer_id": "",
                "preferences": []
            }
        if "escalation_threshold" not in st.session_state:
            st.session_state.escalation_threshold = 0.7
        if "model_confidence" not in st.session_state:
            st.session_state.model_confidence = 0.0

    @st.cache_resource
    def preprocess_text(_self, text):
        """Enhanced text preprocessing with better cleaning"""
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
        
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        
        return ' '.join(tokens)

    @st.cache_data
    def load_data(_self):
        """Load sample data for demonstration"""
        data1 = pd.read_excel('NLP Project A dataset.xlsx', sheet_name='data1')
        data2 = pd.read_excel('NLP Project A dataset.xlsx', sheet_name='data2')
        data3 = pd.read_excel('NLP Project A dataset.xlsx', sheet_name='data3')
        df = pd.concat([data1, data2, data3], ignore_index=True)
        df.dropna(inplace=True)
        df.drop_duplicates(subset=['utterance'], inplace=True)
        return df

    @st.cache_data
    def prepare_data(_self, df):
        """Prepare data for model training"""
        df['processed'] = df['utterance'].apply(_self.preprocess_text)
        
        tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), min_df=1, max_df=0.9)
        X_tfidf = tfidf.fit_transform(df['processed']).toarray()
        
        tokenizer = Tokenizer(num_words=1000, oov_token='<OOV>')
        tokenizer.fit_on_texts(df['processed'])
        X_seq = tokenizer.texts_to_sequences(df['processed'])
        X_pad = pad_sequences(X_seq, maxlen=50, padding='post', truncating='post')
        
        le = LabelEncoder()
        y = le.fit_transform(df['category'])
        
        return X_tfidf, X_pad, y, tfidf, tokenizer, le

    def train_enhanced_xgboost(self, X, y):
        """Train XGBoost with updated parameters"""
        model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss'
        )
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model.fit(X_train, y_train)
        
        try:
            joblib.dump(model, "xgboost_model.pkl")
        except Exception as e:
            st.error(f"Failed to save XGBoost model: {e}")
        
        return model

    def train_enhanced_lstm(self, X, y, num_classes, vocab_size):
        """Train LSTM with simplified architecture"""
        model = Sequential([
            Embedding(vocab_size, 50, input_length=X.shape[1]),
            LSTM(64, dropout=0.3, return_sequences=True),
            LSTM(32, dropout=0.3),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=5,
            batch_size=16,
            verbose=0
        )
        
        try:
            model.save("lstm_model.h5")
        except Exception as e:
            st.error(f"Failed to save LSTM model: {e}")
        
        return model

    def analyze_sentiment(self, text):
        """Analyze sentiment of user input"""
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity
        
        if sentiment > 0.1:
            return "positive", sentiment
        elif sentiment < -0.1:
            return "negative", sentiment
        else:
            return "neutral", sentiment

    def ensemble_predict_with_confidence(self, text):
        """Ensemble prediction with confidence scores"""
        processed_text = self.preprocess_text(text)
        
        x_tfidf = self.tfidf.transform([processed_text]).toarray()
        xgb_proba = self.xgb_model.predict_proba(x_tfidf)[0]
        
        x_seq = self.tokenizer.texts_to_sequences([processed_text])
        x_pad = pad_sequences(x_seq, maxlen=50)
        lstm_proba = self.lstm_model.predict(x_pad, verbose=0)[0]
        
        ensemble_proba = 0.6 * xgb_proba + 0.4 * lstm_proba
        
        predicted_class_idx = np.argmax(ensemble_proba)
        confidence = float(ensemble_proba[predicted_class_idx])
        predicted_category = self.le.inverse_transform([predicted_class_idx])[0]
        
        return predicted_category, confidence, ensemble_proba

    def generate_smart_reply(self, category, sentiment, confidence, user_input):
        """Generate contextual replies"""
        base_responses = {
            "refund": {
                "positive": "I'll be happy to help with your refund request!",
                "neutral": "I understand you're requesting a refund. Let me assist you.",
                "negative": "I'm sorry you're having issues. I'll prioritize your refund request."
            },
            "payment": {
                "positive": "I'll help resolve your payment query quickly.",
                "neutral": "Your payment-related query is being processed.",
                "negative": "I understand your payment concerns. Let me address this immediately."
            },
            "delivery": {
                "positive": "I'll check your delivery status right away!",
                "neutral": "Let me track your delivery for you.",
                "negative": "I'm sorry for any delivery issues. Let me investigate this."
            },
            "account": {
                "positive": "I'll help with your account settings.",
                "neutral": "I can assist with your account-related query.",
                "negative": "I understand your account concerns. Let me help resolve this."
            }
        }
        
        category_responses = base_responses.get(category.lower(), {
            "positive": "Thank you for contacting us! I'll help you right away.",
            "neutral": "I'll assist with your request.",
            "negative": "I'm sorry you're experiencing issues. Let me help you."
        })
        
        response = category_responses.get(sentiment, category_responses["neutral"])
        
        if confidence < 0.6:
            response += "\n\n*I'm not entirely sure about the category of your request. A human agent will review this.*"
        elif confidence > 0.9:
            response += f"\n\n*I'm {confidence:.0%} confident I understand your request correctly.*"
        
        return response

    def load_models_and_preprocessors(self):
        """Load or train models"""
        with st.spinner("Loading AI models..."):
            self.df = self.load_data()
            X_tfidf, X_pad, y, self.tfidf, self.tokenizer, self.le = self.prepare_data(self.df)
            
            if os.path.exists("xgboost_model.pkl") and os.path.exists("lstm_model.h5"):
                try:
                    self.xgb_model = joblib.load("xgboost_model.pkl")
                    self.lstm_model = load_model("lstm_model.h5")
                    # st.success("Pre-trained models loaded successfully!")
                except Exception as e:
                    st.error(f"Failed to load models: {e}")
                    st.info("Training new models...")
                    self.xgb_model = self.train_enhanced_xgboost(X_tfidf, y)
                    self.lstm_model = self.train_enhanced_lstm(X_pad, y, len(self.le.classes_), 1000)
            else:
                with st.spinner("Training new models..."):
                    self.xgb_model = self.train_enhanced_xgboost(X_tfidf, y)
                    self.lstm_model = self.train_enhanced_lstm(X_pad, y, len(self.le.classes_), 1000)
                    st.success("New models trained successfully!")

    def create_analytics_dashboard(self):
        """Create analytics dashboard"""
        st.header("📊 Analytics Dashboard")
        
        stats = st.session_state.conversation_stats
        
        if stats["total_queries"] == 0:
            st.info("No conversations yet. Start chatting to see analytics!")
            return
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Queries", stats["total_queries"])
        
        # with col2:
        #     avg_confidence = np.mean([conf for conf in st.session_state.get("confidence_scores", [0.8])])
        #     st.metric("Avg Confidence", f"{avg_confidence:.2%}")
        
        with col2:
            if stats["sentiment_scores"]:
                avg_sentiment = np.mean(stats["sentiment_scores"])
                sentiment_label = "😊 Positive" if avg_sentiment > 0 else "😞 Negative" if avg_sentiment < 0 else "😐 Neutral"
                st.metric("Avg Sentiment", sentiment_label)
            else:
                st.metric("Avg Sentiment", "N/A")
        
        with col3:
            if stats["response_times"]:
                avg_response_time = np.mean(stats["response_times"])
                st.metric("Avg Response Time", f"{avg_response_time:.2f}s")
            else:
                st.metric("Avg Response Time", "N/A")
        
        if stats["categories_count"]:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Query Categories Distribution")
                categories = list(stats["categories_count"].keys())
                counts = list(stats["categories_count"].values())
                
                fig = px.pie(values=counts, names=categories, title="Query Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Category Trends")
                fig = px.bar(x=categories, y=counts, title="Queries by Category")
                fig.update_layout(xaxis_title="Category", yaxis_title="Count")
                st.plotly_chart(fig, use_container_width=True)

    def create_model_performance_tab(self):
        """Create model performance analysis tab"""
        st.header("🔬 Model Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("XGBoost Model")
            if hasattr(self, 'xgb_model'):
                st.write("✅ Model loaded successfully")
                st.write(f"Number of features: {self.xgb_model.n_features_in_}")
                st.write(f"Number of classes: {len(self.le.classes_)}")
        
        with col2:
            st.subheader("LSTM Model")
            if hasattr(self, 'lstm_model'):
                st.write("✅ Model loaded successfully")
                st.write(f"Model parameters: {self.lstm_model.count_params():,}")
                st.write(f"Input shape: {self.lstm_model.input_shape}")
        
        st.subheader("Test Model Prediction")
        test_text = st.text_input("Enter text to test model:", 
                                  placeholder="e.g., I want to return my order")
        
        if test_text:
            category, confidence, probabilities = self.ensemble_predict_with_confidence(test_text)
            sentiment, sentiment_score = self.analyze_sentiment(test_text)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Predicted Category", category)
            with col2:
                st.metric("Confidence", f"{confidence:.2%}")
            with col3:
                st.metric("Sentiment", f"{sentiment} ({sentiment_score:.2f})")
            
            st.subheader("Prediction Probabilities")
            prob_df = pd.DataFrame({
                'Category': self.le.classes_,
                'Probability': probabilities
            }).sort_values('Probability', ascending=False)
            
            fig = px.bar(prob_df, x='Category', y='Probability', 
                        title="Category Probability Distribution")
            st.plotly_chart(fig, use_container_width=True)

    def create_settings_tab(self):
        """Create settings and configuration tab"""
        st.header("⚙️ Settings & Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("User Profile")
            name = st.text_input("Name", value=st.session_state.user_profile["name"])
            email = st.text_input("Email", value=st.session_state.user_profile["email"])
            customer_id = st.text_input("Customer ID", value=st.session_state.user_profile["customer_id"])
            
            if st.button("Update Profile"):
                st.session_state.user_profile.update({
                    "name": name,
                    "email": email,
                    "customer_id": customer_id
                })
                st.success("Profile updated successfully!")
        
        with col2:
            st.subheader("Chatbot Settings")
            escalation_threshold = st.slider(
                "Escalation Threshold",
                0.1, 1.0, st.session_state.escalation_threshold, 0.1
            )
            st.session_state.escalation_threshold = escalation_threshold
            
            response_style = st.selectbox(
                "Response Style",
                ["Professional", "Friendly", "Empathetic", "Concise"]
            )
            
            st.subheader("Features")
            st.checkbox("Enable Sentiment Analysis", True)
            st.checkbox("Enable Analytics", True)
            st.checkbox("Enable Auto-escalation", True)
        
        st.subheader("Data Management")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export Chat History"):
                chat_data = {
                    "chat_history": st.session_state.chat_history,
                    "stats": st.session_state.conversation_stats
                }
                st.download_button(
                    "Download Chat Data (JSON)",
                    json.dumps(chat_data, indent=2),
                    "chat_history.json",
                    "application/json"
                )
        
        with col2:
            if st.button("Clear All Data"):
                if st.button("Confirm Clear All"):
                    st.session_state.chat_history = []
                    st.session_state.conversation_stats = {
                        "total_queries": 0,
                        "categories_count": {},
                        "sentiment_scores": [],
                        "response_times": [],
                        "satisfaction_ratings": []
                    }
                    st.success("All data cleared!")

    def main_chat_interface(self):
        """Main chat interface"""
        st.header("💬 Customer Support Chat")
        
        chat_container = st.container()
        
        with chat_container:
            for i, chat_item in enumerate(st.session_state.chat_history):
                user_msg, category, bot_response, timestamp, confidence, sentiment = chat_item[:6]
                st.chat_message("user").write(user_msg)
                
                with st.chat_message("assistant"):
                    st.write(bot_response)
                    
                    with st.expander("View Details"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"**Category:** {category}")
                        # with col2:
                        #     st.write(f"**Confidence:** {confidence:.2%}")
                        with col2:
                            st.write(f"**Sentiment:** {sentiment}")
                        with col3:
                            st.write(f"**Time:** {timestamp}")
                        
                        rating = st.radio(
                            f"Rate this response:",
                            ["😡 Poor", "😐 Okay", "😊 Good", "🤩 Excellent"],
                            key=f"rating_{i}",
                            horizontal=True
                        )
                        
                        if rating and len(chat_item) == 6:
                            st.session_state.chat_history[i] = (*chat_item, rating)
                            st.session_state.conversation_stats["satisfaction_ratings"].append(rating)
        
        user_input = st.chat_input("Type your message here...")
        
        if user_input:
            start_time = time.time()
            st.chat_message("user").write(user_input)
            
            category, confidence, _ = self.ensemble_predict_with_confidence(user_input)
            sentiment, sentiment_score = self.analyze_sentiment(user_input)
            
            response = self.generate_smart_reply(category, sentiment, confidence, user_input)
            
            # if confidence < st.session_state.escalation_threshold:
            #     response += "\n\n🚨 **Escalated to human agent** - Low confidence in automated response."
            
            end_time = time.time()
            response_time = end_time - start_time
            
            with st.chat_message("assistant"):
                st.write(response)
                
                with st.expander("Response Details", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**Category:** {category}")
                    with col2:
                        st.write(f"**Confidence:** {confidence:.2%}")
                    with col3:
                        st.write(f"**Sentiment:** {sentiment}")
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            st.session_state.chat_history.append((
                user_input, category, response, timestamp, confidence, sentiment
            ))
            
            stats = st.session_state.conversation_stats
            stats["total_queries"] += 1
            stats["categories_count"][category] = stats["categories_count"].get(category, 0) + 1
            stats["sentiment_scores"].append(sentiment_score)
            stats["response_times"].append(response_time)
            
            if "confidence_scores" not in st.session_state:
                st.session_state.confidence_scores = []
            st.session_state.confidence_scores.append(confidence)
            
            st.rerun()

def main():
    """Main application function"""
    st.title("🤖 Smart Customer Support AI")
    st.markdown("---")
    
    chatbot = CustomerSupportChatbot()
    
    with st.sidebar:
        st.title("Navigation")
        tab = st.radio(
            "Choose a section:",
            ["💬 Chat", "📊 Analytics", "🔬 Model Performance", "⚙️ Settings"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("### Quick Stats")
        if st.session_state.conversation_stats["total_queries"] > 0:
            st.metric("Total Queries", st.session_state.conversation_stats["total_queries"])
            if st.session_state.conversation_stats["categories_count"]:
                most_common = max(st.session_state.conversation_stats["categories_count"].items(), key=lambda x: x[1])
                st.metric("Most Common Category", most_common[0])
        else:
            st.info("Start chatting to see stats!")
    
    if tab == "💬 Chat":
        chatbot.main_chat_interface()
    elif tab == "📊 Analytics":
        chatbot.create_analytics_dashboard()
    elif tab == "🔬 Model Performance":
        chatbot.create_model_performance_tab()
    elif tab == "⚙️ Settings":
        chatbot.create_settings_tab()

if __name__ == "__main__":
    main()
