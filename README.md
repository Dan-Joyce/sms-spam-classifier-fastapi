SMS Spam Classifier API (FastAPI + ML + Docker)
- Overview

This project implements an end-to-end machine learning pipeline for SMS spam detection, including:

Data preprocessing

Text vectorization (TF-IDF)

Model training (Logistic Regression)

API deployment with FastAPI

Containerization with Docker

The goal is to build a reproducible and deployable ML system, not just a notebook experiment.

- Problem Statement

Classify SMS messages into:

ham (legitimate message)

spam (unwanted or malicious message)

- Project Structure
sms-spam-classifier-fastapi/
│
├── app/                # FastAPI app
│   └── main.py
│
├── data/               # Dataset
│   └── spam.csv
│
├── models/             # Saved model + vectorizer
│   ├── model.joblib
│   └── vectorizer.joblib
│
├── src/                # ML pipeline
│   ├── preprocess.py
│   └── train.py
│
├── tests/              # Tests
├── requirements.txt
├── Dockerfile
└── README.md
- Installation (Local)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
- Train the Model
python src/train.py

Output:

trained model → models/model.joblib

vectorizer → models/vectorizer.joblib

- Run the API (Local)
uvicorn app.main:app --reload

Open:

http://127.0.0.1:8000/docs

Test example:

Input: "Win money now!!!"
Output: "spam"
- Run with Docker
Build image
docker build -t spam-api .
Run container
docker run -p 8000:8000 spam-api

Access API:

http://127.0.0.1:8000/docs
- Model Details

Task: Binary text classification

Model: Logistic Regression

Features: TF-IDF vectorization

Preprocessing:

Lowercasing

URL removal

Number normalization

Special character cleaning

- Evaluation

Example metrics:

Accuracy: 1.00 (toy dataset)

Precision / Recall / F1-score

- Note: Current dataset is small for demonstration purposes.
Model performance would be improved using a real dataset (e.g., SMS Spam Collection).

- Reproducibility

Environment managed via requirements.txt

Deterministic pipeline (no randomness in preprocessing)

Docker ensures consistent deployment

- Testing

Example test:

python test_preprocess.py

Validates:

data loading

text cleaning pipeline

- Deployment Features

FastAPI REST API

Docker containerization

Portable and reproducible ML system

- Future Improvements

Use real-world dataset (UCI SMS Spam Collection)

Add model calibration / confidence score

Implement CI/CD pipeline

Add monitoring (latency, errors)

Deploy to cloud (AWS / Azure)

- Author

Bertina DONFACK
Master 2 – Artificial Intelligence & Data

📜 License

MIT License