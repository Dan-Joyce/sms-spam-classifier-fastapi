# SMS Spam Classifier API

This project implements an end-to-end machine learning pipeline for **SMS spam detection**, including preprocessing, model training, API inference with FastAPI, and Docker containerization.

## Project Overview

The objective is to classify SMS messages into two categories:

- **ham**: legitimate message
- **spam**: unwanted or malicious message

This project was built as a reproducible and deployable ML system rather than a notebook-only experiment.

## Project Structure

```bash
sms-spam-classifier-fastapi/
│
├── app/
│   └── main.py
├── data/
│   └── sms.tsv
├── models/
│   ├── model.joblib
│   └── vectorizer.joblib
├── src/
│   ├── preprocess.py
│   └── train.py
├── tests/
├── Dockerfile
├── LICENSE
├── README.md
├── requirements.txt
└── test_preprocess.py

Dataset

The project uses the SMS Spam Collection dataset, a public dataset for SMS spam detection.

File used: data/sms.tsv

Labels:

ham

spam

Observed class distribution after loading:

ham: 4499

spam: 595

This shows a class imbalance, which partly explains the lower recall on the spam class.

Data Preprocessing

Implemented in src/preprocess.py:

lowercasing

URL normalization

number normalization

special character removal

whitespace cleanup

duplicate removal

Model Training

Implemented in src/train.py.

Model

TF-IDF vectorization

Logistic Regression

Training command
python src/train.py
Output

The training script saves:

models/model.joblib

models/vectorizer.joblib

Evaluation

Example result on the real dataset:

Accuracy: 0.97

Spam precision: 0.97

Spam recall: 0.78

Spam F1-score: 0.87

This indicates strong overall performance, with lower recall on spam due in part to dataset imbalance.

Run the API locally

Start the API with:

uvicorn app.main:app --reload

Then open:

http://127.0.0.1:8000/docs

Example query:

Input: Win money now!!!

Output: spam

Docker
Build the image
docker build -t spam-api .
Run the container
docker run -p 8000:8000 spam-api

Then access:

http://127.0.0.1:8000/docs
Reproducibility

The environment is defined in:

requirements.txt

The project can be executed locally or inside Docker for reproducible deployment.

Testing

Basic preprocessing validation is available in:

test_preprocess.py

Run with:

python test_preprocess.py
License

This project is distributed under the MIT License.

Author

Bertina DONFACK