import os
import pandas as pd
import random

# ---------------- CONFIG ----------------
departments = ["Housekeeping", "Reception", "F&B"]
sentiments = ["pos", "neg"]

target_size = 200

# ---------------- BASE ----------------

hk = {
    "pos": ["camera pulitissima", "tutto ok con la stanza", "camera davvero in ordine", "buona pulizia generale"],
    "neg": ["camera sporca", "non era pulita bene", "pulizia scarsa", "camera lasciata male"]
}

rec = {
    "pos": ["staff gentile", "personale molto disponibile", "accoglienza buona", "reception veloce"],
    "neg": ["personale scortese", "attesa lunga", "reception lenta", "servizio non buono"]
}

fb = {
    "pos": ["colazione buona", "cibo ok", "buffet soddisfacente", "tutto abbastanza buono"],
    "neg": ["colazione scarsa", "cibo freddo", "poco assortimento", "qualità bassa"]
}

# ---------------- TEMPLATES (NO REPARTO NEL TESTO!) ----------------

templates = [
    "la stanza è {base}",
    "{base} durante il soggiorno",
    "nel complesso {base}",
    "durante il soggiorno {base}",
    "{base} direi"
]

# ---------------- NOISE ----------------

typos = {
    "camera": ["camra", "cmera", "cameraa"],
    "pulita": ["pulitta", "pulitaa"],
    "personale": ["persoale", "personalle"],
    "colazione": ["colazzione", "colazionee"],
    "servizio": ["servizzio", "servizioo"]
}

case_noise = [
    lambda x: x,
    lambda x: x.upper(),
    lambda x: x.capitalize()
]

# ---------------- BASE PICK ----------------

def get_base(dept, sentiment):
    if dept == "Housekeeping":
        return random.choice(hk[sentiment])
    elif dept == "Reception":
        return random.choice(rec[sentiment])
    else:
        return random.choice(fb[sentiment])

# ---------------- TYPO NOISE ----------------

def add_typos(text):
    words = text.split()
    new_words = []

    for w in words:
        lw = w.lower()
        if lw in typos and random.random() < 0.25:
            new_words.append(random.choice(typos[lw]))
        else:
            new_words.append(w)

    return " ".join(new_words)

# ---------------- GENERATION ----------------

data = []
seen = set()

while len(data) < target_size:

    dept = random.choice(departments)
    sent = random.choices(sentiments, weights=[0.7, 0.3])[0]

    base = get_base(dept, sent)
    template = random.choice(templates)

    text = template.format(base=base)

    # case noise
    text = random.choice(case_noise)(text)

    # typo noise
    text = add_typos(text)

    key = text.lower().strip()

    if key in seen:
        continue

    seen.add(key)

    data.append({
        "text": text,
        "department": dept,
        "sentiment": sent
    })

# ---------------- DATAFRAME ----------------

df = pd.DataFrame(data)

# ---------------- ID FIRST COLUMN ----------------

df = df.reset_index(drop=True)
df.insert(0, "id", df.index)

# ---------------- SAVE ----------------

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
DATA_PATH = os.path.join(BASE_DIR, "data", "reviews_noisy.csv")

os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

df.to_csv(DATA_PATH, index=False, encoding="utf-8-sig")

print(f"✅ Generated {len(df)} NOISY reviews (NO LEAKAGE)")
print("📁 Saved in:", DATA_PATH)