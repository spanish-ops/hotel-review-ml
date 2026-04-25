import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

INPUT_FILE = os.path.join(DATA_DIR, "raw_real_reviews.txt")
OUTPUT_FILE = os.path.join(DATA_DIR, "reviews_real.csv")

reviews = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    text = line.strip()
    
    if text:  # evita righe vuote
        reviews.append({
            "id": i,
            "text": text,
            "department": "",   # da etichettare dopo
            "sentiment": ""     # da etichettare dopo
        })

df = pd.DataFrame(reviews)

df.to_csv(OUTPUT_FILE, index=False)

print("CSV creato da file TXT!")