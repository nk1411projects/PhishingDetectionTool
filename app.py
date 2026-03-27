from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load trained ML model and vectorizer
model = joblib.load("models/phishing_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

def check_phishing(url):
    url_vectorized = vectorizer.transform([url])
    prediction = model.predict(url_vectorized)[0]
    return "Phishing" if prediction == 1 else "Legitimate"

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    submitted_url = ""
    if request.method == "POST":
        submitted_url = request.form["url"].strip()
        if submitted_url:
            result = check_phishing(submitted_url)
    return render_template("index.html", result=result, submitted_url=submitted_url)

if __name__ == "__main__":
    app.run(debug=True)
