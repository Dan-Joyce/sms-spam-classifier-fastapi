from src.preprocess import load_and_clean_data

df = load_and_clean_data("data/sms.tsv")

print(df["label"].value_counts())