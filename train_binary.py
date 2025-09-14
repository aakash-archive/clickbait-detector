# train_binary.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
from utils import preprocess_text

# Load dataset (Webis Clickbait 2017 preprocessed CSV)
# Expected columns: "headline", "label" (0=neutral, 1=clickbait)
data = pd.read_csv("data/clickbait_data.csv")

# Preprocess
data["headline"] = data["headline"].astype(str).apply(preprocess_text)

X = data["headline"]
y = data["clickbait"]

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

# Train Logistic Regression
model = LogisticRegression(max_iter=300)
model.fit(X_tfidf, y)

# Save
joblib.dump(model, "models/binary_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("✅ Binary model trained and saved in models/")
