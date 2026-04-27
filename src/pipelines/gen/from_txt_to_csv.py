import pandas as pd
import os
import csv

# ---------------- ROOT PROGETTO ----------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
DATA_DIR = os.path.join(BASE_DIR, "data")

INPUT_FILE = os.path.join(DATA_DIR, "raw_real_reviews.txt")
OUTPUT_FILE = os.path.join(DATA_DIR, "reviews_real.csv")

# ---------------- CHECK FILE ----------------
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"File non trovato: {INPUT_FILE}")

# ---------------- LOAD TXT ----------------
reviews = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        text = line.strip()

        if text:
            reviews.append({
                "id": i,
                "text": text,
                "department": "",
                "sentiment": ""
            })

# ---------------- DATAFRAME ----------------
df = pd.DataFrame(reviews)

# ---------------- FIX IMPORTANTE ----------------
# CSV safe (gestisce virgole, testi lunghi, ecc.)
df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig",
    quoting=csv.QUOTE_ALL
)

print(f"✅ CSV creato correttamente!")
print(f"📁 Salvato in: {OUTPUT_FILE}")
print(f"📊 Totale recensioni: {len(df)}")