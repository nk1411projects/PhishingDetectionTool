import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
import os

# Load dataset
df = pd.read_csv("dataset/phishing_urls.csv")

# Debugging: Print column names
print("Columns in dataset:", df.columns)

# Ensure correct column names
if "url" not in df.columns or "status" not in df.columns:
    raise ValueError("Dataset must have 'url' and 'status' columns.")

# Convert labels to numerical format (phishing = 1, legitimate = 0)
df["status"] = df["status"].map({"phishing": 1, "legitimate": 0})

# Extract features and labels
X = df["url"]
y = df["status"]

# Convert URLs to numerical format using CountVectorizer
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Create models directory if not exists
models_dir = "models"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Save model and vectorizer
joblib.dump(model, os.path.join(models_dir, "phishing_model.pkl"))
joblib.dump(vectorizer, os.path.join(models_dir, "vectorizer.pkl"))

print("✅ Model training complete! Model and vectorizer saved successfully.")
