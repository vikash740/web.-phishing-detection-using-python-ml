# phishing_detection_app.py
# 📌 Phishing Website Detection Web App (Streamlit)
# Author: Your Name | Date: 2025

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.parse import urlparse

# -----------------------------
# 🎨 Streamlit App Layout
# -----------------------------
st.set_page_config(page_title="Phishing Website Detection", layout="wide")
st.title("🛡 Phishing Website Detection using Machine Learning")
st.write("Upload your dataset and check if a website is phishing or legit!")

# -----------------------------
# 📂 STEP 1: File Upload
# -----------------------------
uploaded_file = st.file_uploader("📂 Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset Loaded Successfully!")
    st.write("### 🔎 Preview of Dataset")
    st.dataframe(df.head())

    # -----------------------------
    # 🧹 STEP 2: Data Preprocessing
    # -----------------------------
    st.subheader("🔄 Data Preprocessing")
    if 'Result' not in df.columns:
        st.error("❌ No 'Result' column found in dataset! Please include target column.")
    else:
        X = df.drop('Result', axis=1)
        y = df['Result']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        st.write(f"✅ Data Split Complete! *Training Samples:* {len(X_train)}, *Testing Samples:* {len(X_test)}")

        # -----------------------------
        # 🤖 STEP 3: Train Model
        # -----------------------------
        st.subheader("🤖 Training Random Forest Model")
        n_estimators = st.slider("Number of Trees (n_estimators)", 10, 300, 100, step=10)
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        st.success("✅ Model Training Complete!")

        # -----------------------------
        # 📊 STEP 4: Evaluate Model
        # -----------------------------
        st.subheader("📊 Model Evaluation")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.metric(label="🎯 Model Accuracy", value=f"{accuracy*100:.2f}%")

        st.text("Classification Report")
        st.code(classification_report(y_test, y_pred), language="text")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

        # -----------------------------
        # 🔮 STEP 5: Sample Prediction
        # -----------------------------
        st.subheader("🔮 Try a Sample Prediction")
        if st.button("Predict First Sample"):
            sample = X_test.iloc[0].values.reshape(1, -1)
            prediction = model.predict(sample)
            result = "🚨 Phishing Website" if prediction[0] == 1 else "✅ Legit Website"
            st.info(f"Prediction for first test sample: *{result}*")

        # -----------------------------
        # 🌐 STEP 6: URL Based Prediction
        # -----------------------------
        st.subheader("🌐 Predict Using Website URL")

        url = st.text_input("🔗 Enter Website URL")

        # -----------------------------
        # URL → Features Extract Function (Dummy Example)
        # -----------------------------
        def extract_features_from_url(url):
            parsed = urlparse(url)

            features = {
                "url_length": len(url),
                "num_digits": sum(c.isdigit() for c in url),
                "num_special_char": sum(not c.isalnum() for c in url),
                "has_https": 1 if parsed.scheme == "https" else 0,
                "subdomain_count": len(parsed.netloc.split(".")) - 2,
            }

            # Missing features will be filled automatically
            all_features = pd.DataFrame([features])

            # Align with training dataset columns
            for col in X.columns:
                if col not in all_features.columns:
                    all_features[col] = 0  # default value

            return all_features[X.columns]

        if st.button("🔍 Predict URL"):
            if url.strip() == "":
                st.error("⚠️ Please enter a valid URL!")
            else:
                try:
                    url_features = extract_features_from_url(url)
                    prediction = model.predict(url_features)[0]
                    result = "🚨 Phishing Website" if prediction == 1 else "✅ Legit Website"
                    st.success(f"Prediction for URL: **{result}**")
                except Exception as e:
                    st.error(f"Error processing URL: {e}")

else:
    st.warning("👆 Please upload a CSV file to continue.")
