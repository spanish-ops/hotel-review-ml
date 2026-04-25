import pandas as pd
import os
from datetime import datetime

# ---------------- PATH ----------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

CLEAN_PATH = os.path.join(DATA_DIR, "reviews_clean.csv")
NOISY_PATH = os.path.join(DATA_DIR, "reviews_noisy.csv")
REAL_PATH = os.path.join(DATA_DIR, "reviews_real.csv")

OUTPUT_PATH = os.path.join(DATA_DIR, "reviews_dataset.csv")


# ---------------- FUNZIONE SICURA CARICAMENTO ----------------
def load_dataset(path, name):
    if os.path.exists(path):
        print(f"✔ Caricato {name}")
        return pd.read_csv(path)
    else:
        print(f"⚠ {name} NON trovato: {path}")
        return pd.DataFrame()


# ---------------- CARICAMENTO ----------------
clean_df = load_dataset(CLEAN_PATH, "CLEAN")
noisy_df = load_dataset(NOISY_PATH, "NOISY")
real_df = load_dataset(REAL_PATH, "REAL")


# ---------------- CONTROLLI ----------------
if clean_df.empty and noisy_df.empty and real_df.empty:
    raise ValueError("❌ Nessun dataset trovato. Genera prima i file!")


# ---------------- UNIONE ----------------
df = pd.concat([clean_df, noisy_df, real_df], ignore_index=True)

# ---------------- SHUFFLE ----------------
df = df.sample(frac=1, random_state=42).reset_index(drop=True)


# ---------------- INFO UTILI ----------------
print("\n📊 DISTRIBUZIONE DEPARTMENT:")
print(df["department"].value_counts())

print("\n📊 DISTRIBUZIONE SENTIMENT:")
print(df["sentiment"].value_counts())

print("\n📊 DIMENSIONE DATASET:", len(df))


# ---------------- SALVATAGGIO ----------------
os.makedirs(DATA_DIR, exist_ok=True)

df.to_csv(OUTPUT_PATH, index=False)

print(f"\n✅ Dataset finale salvato in: {OUTPUT_PATH}")


# ---------------- EXPORT CON TIMESTAMP (BONUS) ----------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_path = os.path.join(DATA_DIR, f"reviews_final_{timestamp}.csv")

df.to_csv(backup_path, index=False)

print(f"🕒 Backup creato: {backup_path}")