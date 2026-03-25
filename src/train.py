import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from preprocess import load_and_clean_data

def main():
    # 1) Charger les données
    df = load_and_clean_data("data/sms.tsv")

    # 2) Transformer les labels texte en nombres
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

    # 3) Séparer les données en train et test
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label_num"],
        test_size=0.2,
        random_state=42,
        stratify=df["label_num"]
    )

    # 4) Transformer le texte en vecteurs numériques
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 5) Créer et entraîner le modèle
    model = LogisticRegression(max_iter=200)
    model.fit(X_train_vec, y_train)

    # 6) Évaluer le modèle
    y_pred = model.predict(X_test_vec)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    # 7) Sauvegarder le modèle et le vectorizer
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.joblib")
    joblib.dump(vectorizer, "models/vectorizer.joblib")

    print("\nModel and vectorizer saved in models/")


if __name__ == "__main__":
    main()