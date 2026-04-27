import pandas as pd
import os
from datetime import datetime

# ---------------- PATH CORRETTO ----------------
# risale fino alla root del progetto (hotel-review-ml/)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
DATA_DIR = os.path.join(BASE_DIR, "data")

CLEAN_PATH = os.path.join(DATA_DIR, "reviews_clean.csv")
NOISY_PATH = os.path.join(DATA_DIR, "reviews_noisy.csv")
REAL_PATH = os.path.join(DATA_DIR, "reviews_real.csv")

OUTPUT_PATH = os.path.join(DATA_DIR, "reviews_dataset.csv")


# ---------------- LOAD SAFELY ----------------
def load_dataset(path, name):
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"✔ {name} caricato -> {len(df)} righe")
        return df
    else:
        print(f"⚠ {name} NON trovato: {path}")
        return pd.DataFrame()


clean_df = load_dataset(CLEAN_PATH, "CLEAN")
noisy_df = load_dataset(NOISY_PATH, "NOISY")
real_df = load_dataset(REAL_PATH, "REAL")


# ---------------- CHECK ----------------
if clean_df.empty and noisy_df.empty and real_df.empty:
    raise ValueError("❌ Nessun dataset trovato. Genera prima i file!")


# ---------------- MERGE ----------------
df = pd.concat([clean_df, noisy_df, real_df], ignore_index=True)


# ---------------- CLEAN DUPLICATI ----------------
if "text" in df.columns:
    before = len(df)
    df = df.drop_duplicates(subset=["text"])
    print(f"🔁 Duplicati rimossi: {before - len(df)}")


# ---------------- SHUFFLE ----------------
df = df.sample(frac=1, random_state=42).reset_index(drop=True)


# ---------------- DEBUG INFO ----------------
print("\n📦 SHAPE:", df.shape)

if "department" in df.columns:
    print("\n📊 Department distribution:")
    print(df["department"].value_counts())

if "sentiment" in df.columns:
    print("\n📊 Sentiment distribution:")
    print(df["sentiment"].value_counts())


# ---------------- SAVE FINAL ----------------
os.makedirs(DATA_DIR, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print(f"\n✅ Dataset salvato in: {OUTPUT_PATH}")


# ---------------- BACKUP ----------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_path = os.path.join(DATA_DIR, f"reviews_final_{timestamp}.csv")

df.to_csv(backup_path, index=False, encoding="utf-8-sig")

print(f"🕒 Backup creato: {backup_path}")