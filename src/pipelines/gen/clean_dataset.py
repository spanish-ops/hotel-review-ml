import pandas as pd

df = pd.read_csv("../data/reviews_dataset.csv", encoding="latin-1")

# pulizia base
df = df.dropna()
df = df.drop_duplicates(subset=["text", "sentiment", "department"])

# salva correttamente in UTF-8
df.to_csv("../data/reviews_dataset_clean.csv", index=False, encoding="utf-8")

print("Fatto!")