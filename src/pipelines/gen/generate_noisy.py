import os
import pandas as pd
import random

# ---------------- CONFIG ----------------
departments = ["Housekeeping", "Reception", "F&B"]
sentiments = ["pos", "neg"]

target_size = 200

# ---------------- BASE (più colloquiale e diversa dal clean) ----------------

hk = {
    "pos": ["camera pulitissima", "tutto ok con la stanza", "camera davvero in ordine", "buona pulizia generale"],
    "neg": ["camera sporca", "non era pulita bene", "pulizia scarsa", "camera lasciata male"]
}

rec = {
    "pos": ["staff gentile", "personale ok", "accoglienza buona", "reception veloce"],
    "neg": ["personale scortese", "attesa lunga", "reception lenta", "servizio non buono"]
}

fb = {
    "pos": ["colazione buona", "cibo ok", "buffet soddisfacente", "tutto abbastanza buono"],
    "neg": ["colazione scarsa", "cibo freddo", "poco assortimento", "qualità bassa"]
}

# ---------------- TEMPLATE DIVERSI DAL CLEAN ----------------

templates = [
    "la {dept} è {base}",
    "{base} alla {dept}",
    "nel complesso {base}",
    "durante il soggiorno {base}",
    "{base} direi"
]

# ---------------- NOISE ----------------

typos = {
    "camera": ["camra", "cmera", "cameraa"],
    "pulita": ["pulitta", "pulitaa", "pulita"],
    "personale": ["personale", "persoale", "personalle"],
    "colazione": ["colazzione", "colazionee", "colazone"],
    "servizio": ["servizzio", "servizioo", "servizzio"]
}

depart_ambiguity = [
    "",
    " forse housekeeping",
    " reception o forse ristorante",
    " non so se housekeeping o reception",
    " area servizio"
]

case_noise = [
    lambda x: x,
    lambda x: x.upper(),
    lambda x: x.capitalize()
]

# ---------------- FUNZIONI ----------------

def get_base(dept, sentiment):
    if dept == "Housekeeping":
        return random.choice(hk[sentiment])
    elif dept == "Reception":
        return random.choice(rec[sentiment])
    else:
        return random.choice(fb[sentiment])

def add_typos(text):
    for word, variants in typos.items():
        if word in text and random.random() < 0.3:
            text = text.replace(word, random.choice(variants))
    return text

# ---------------- GENERAZIONE ----------------

data = []
seen = set()
i = 0

while len(data) < target_size:

    dept = random.choice(departments)
    sent = random.choices(sentiments, weights=[0.7, 0.3])[0]

    base = get_base(dept, sent)

    template = random.choice(templates)

    text = template.format(dept=dept, base=base)

    # noise 1: maiuscole/minuscole
    text = random.choice(case_noise)(text)

    # noise 2: ambiguità reparto
    if random.random() < 0.25:
        text += random.choice(depart_ambiguity)

    # noise 3: errori ortografici leggeri
    text = add_typos(text)

    key = text.lower().strip()

    if key in seen:
        continue

    seen.add(key)

    data.append({
        "id": i,
        "text": text,
        "department": dept,
        "sentiment": sent
    })

    i += 1

# ---------------- SAVE ----------------

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
DATA_PATH = os.path.join(BASE_DIR, "data", "reviews_noisy.csv")

os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

df = pd.DataFrame(data)
df.to_csv(DATA_PATH, index=False, encoding="utf-8-sig")

print(f"✅ Generated {len(df)} NOISY reviews")
print("📁 Saved in:", DATA_PATH)