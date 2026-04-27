import pandas as pd

df = pd.read_csv("../data/dataset.csv")

# Rimuovi duplicati
df = df.drop_duplicates(subset=["text"])

# Rimuovi righe con valori mancanti
df = df.dropna()

# Sistema encoding (opzionale veloce)
df["text"] = df["text"].str.replace("Ã ", "à")
df["text"] = df["text"].str.replace("Ã¨", "è")

# Reset index
df = df.reset_index(drop=True)

# Salva pulito
df.to_csv("../data/dataset_clean.csv", index=False)

print("Dataset pulito salvato!")