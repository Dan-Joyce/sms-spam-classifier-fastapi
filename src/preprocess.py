import re
import pandas as pd

def clean_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"http\S+|www\S+", " URL ", text)
    text = re.sub(r"\d+", " NUM ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_and_clean_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", names=["label", "text"])
    df = df.rename(columns={df.columns[0]: "label", df.columns[1]: "text"})
    df = df.dropna(subset=["label", "text"])
    df["text"] = df["text"].astype(str).apply(clean_text)
    df = df.drop_duplicates()
    return df