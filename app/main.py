from fastapi import FastAPI
import joblib

app = FastAPI()

# Charger le modèle et le vectorizer
model = joblib.load("models/model.joblib")
vectorizer = joblib.load("models/vectorizer.joblib")


@app.get("/")
def home():
    return {"message": "Spam Classifier API is running"}


@app.get("/predict")
def predict(text: str):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]

    label = "spam" if prediction == 1 else "ham"

    return {
        "input": text,
        "prediction": label
    }