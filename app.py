# app.py
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from src.train_load_model import train_load_model
from src.predict import predict_sentiment
from src.preprocess import preprocess

st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")


nb, tfidf, le, metrics = train_load_model()

st.title("📊 SENTIMENT ANALYSIS DASHBOARD")
st.write("Analyze customer reviews with Naive Bayes")


review_input = st.text_area("Enter your review:")

if st.button("Predict Sentiment") and review_input.strip():
    label, emoji, stars = predict_sentiment(review_input, nb, tfidf, le, preprocess)
    st.subheader(f"Prediction: {label} {emoji}")
    st.write(f"⭐ Stars: {stars}")


st.subheader("Model Metrics")

if metrics:
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{metrics['accuracy']:.2f}")
    col2.metric("F1 Score (Macro)", f"{metrics['f1_macro']:.2f}")
    col3.metric("F1 Score (Weighted)", f"{metrics['f1_weighted']:.2f}")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(metrics['conf_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)


    st.subheader("Class Distribution")
    import numpy as np
    counts = np.sum(metrics['conf_matrix'], axis=1)
    class_df = {"Class": le.classes_, "Count": counts}
    fig2 = px.bar(class_df, x="Class", y="Count", color="Class",
                  text="Count", color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig2, use_container_width=True)
