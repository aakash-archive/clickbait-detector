import streamlit as st
import joblib
import pandas as pd
import altair as alt
from utils import preprocess_text, get_headline_from_url
from transformers import pipeline

# Load models
binary_model = joblib.load("models/binary_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")
bert_classifier = pipeline("text-classification", model="models/bert_emotion", tokenizer="models/bert_emotion")

st.set_page_config(page_title="Clickbait Detector", page_icon="📰", layout="centered")
st.title("📰 Emotionally Manipulative Clickbait Detector")
st.markdown("Detect **Clickbait** and underlying **Emotion** in headlines or article URLs.")

# Session state for history
if "history" not in st.session_state:
    st.session_state.history = []

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["🔤 Enter Headline", "🌐 Enter URL", "📂 Upload CSV"])

headline = None
results = []

# Tab 1: Manual headline
with tab1:
    headline = st.text_area("Enter a headline:", "")
    if st.button("🔍 Detect", key="detect_headline"):
        if not headline.strip():
            st.warning("⚠️ Please enter a headline.")
        else:
            processed = preprocess_text(headline)
            features = vectorizer.transform([processed])
            pred_binary = binary_model.predict(features)[0]
            prob_binary = binary_model.predict_proba(features).max()

            if pred_binary == 1:
                emotion = bert_classifier(headline)[0]
                st.error(f"🚨 Manipulative Clickbait Detected! (Conf: {prob_binary:.2f})")
                st.markdown(f"**Emotion Exploited:** :red[{emotion['label']}]  **Confidence:** {emotion['score']:.2f}")
                result = {"headline": headline, "clickbait": 1, "prob": prob_binary, "emotion": emotion['label']}
            else:
                st.success(f"✅ Neutral Headline (Conf: {prob_binary:.2f})")
                result = {"headline": headline, "clickbait": 0, "prob": prob_binary, "emotion": "neutral"}
            st.session_state.history.insert(0, result)

# Tab 2: URL headline fetch
with tab2:
    url = st.text_input("Enter a news article URL:", "")
    if st.button("Fetch Headline"):
        headline = get_headline_from_url(url)
        if headline:
            st.success(f"Extracted Headline: *{headline}*")
        else:
            st.error("Could not fetch headline. Try another URL.")

# Tab 3: Batch CSV upload
with tab3:
    file = st.file_uploader("Upload CSV with 'headline' column", type="csv")
    if file is not None:
        df = pd.read_csv(file)
        if "headline" not in df.columns:
            st.error("CSV must have a column named 'headline'")
        else:
            for text in df["headline"].astype(str).tolist():
                processed = preprocess_text(text)
                features = vectorizer.transform([processed])
                pred_binary = binary_model.predict(features)[0]
                prob_binary = binary_model.predict_proba(features).max()
                if pred_binary == 1:
                    emotion = bert_classifier(text)[0]
                    results.append({"headline": text, "clickbait": 1, "prob": prob_binary, "emotion": emotion['label']})
                else:
                    results.append({"headline": text, "clickbait": 0, "prob": prob_binary, "emotion": "neutral"})

            results_df = pd.DataFrame(results)
            st.dataframe(results_df)

            # Pie chart: clickbait vs neutral
            pie = alt.Chart(results_df).mark_arc().encode(
                theta="count():Q", color="clickbait:N"
            )
            st.altair_chart(pie, use_container_width=True)

            # Bar chart: emotions
            bar = alt.Chart(results_df).mark_bar().encode(
                x="emotion:N", y="count():Q", color="emotion:N"
            )
            st.altair_chart(bar, use_container_width=True)

# History of last 5 predictions
st.markdown("---")
st.subheader("📝 Last 5 Predictions")
if st.session_state.history:
    hist_df = pd.DataFrame(st.session_state.history[:5])
    st.table(hist_df)
else:
    st.caption("No predictions yet.")

st.markdown("---")
st.caption("Built with using Streamlit, Scikit-learn, and HuggingFace Transformers")