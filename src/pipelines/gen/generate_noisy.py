import os
import pandas as pd
import random

# ---------------- CONFIG ----------------
departments = ["Housekeeping", "Reception", "F&B"]
sentiments = ["pos", "neg"]

target_size = 200

# ---------------- BASE ----------------

hk = {
    "pos": [
        "camera molto pulita",
        "stanza in ordine",
        "ambiente curato",
        "pulizia eccellente"
    ],
    "neg": [
        "camera sporca",
        "pulizia scarsa",
        "stanza trascurata",
        "ambiente poco pulito"
    ]
}

rec = {
    "pos": [
        "staff gentile",
        "personale disponibile",
        "accoglienza cordiale",
        "servizio veloce"
    ],
    "neg": [
        "personale scortese",
        "attesa lunga",
        "servizio lento",
        "accoglienza fredda"
    ]
}

fb = {
    "pos": [
        "colazione buona",
        "cibo di qualità",
        "buffet abbondante",
        "piatti gustosi"
    ],
    "neg": [
        "colazione scarsa",
        "cibo freddo",
        "poca scelta",
        "qualità bassa"
    ]
}

# ---------------- TEMPLATES ----------------

templates = [
    "{base}",
    "{base} durante il soggiorno",
    "nel complesso {base}",
    "direi che {base}",
    "{base}, ma migliorabile"
]

# ---------------- NOISE (MOLTO PIÙ LEGGERO) ----------------

# typo più realistici (pochi!)
typos = {
    "camera": ["camra"],
    "personale": ["personle"],
    "colazione": ["colazionee"]
}

def add_typos(text):
    words = text.split()
    new_words = []

    for w in words:
        lw = w.lower()
        # 🔴 solo 10% probabilità
        if lw in typos and random.random() < 0.1:
            new_words.append(random.choice(typos[lw]))
        else:
            new_words.append(w)

    return " ".join(new_words)

# case noise più realistico
def add_case_noise(text):
    r = random.random()

    if r < 0.8:
        return text  # normale
    elif r < 0.9:
        return text.capitalize()
    else:
        return text.upper()  # raro

# piccole variazioni realistiche
extra_phrases = [
    "",
    "direi",
    "nel complesso",
    "tutto sommato"
]

def add_extra(text):
    extra = random.choice(extra_phrases)
    if extra:
        return f"{text} {extra}"
    return text

# ---------------- BASE PICK ----------------

def get_base(dept, sentiment):
    if dept == "Housekeeping":
        return random.choice(hk[sentiment])
    elif dept == "Reception":
        return random.choice(rec[sentiment])
    else:
        return random.choice(fb[sentiment])

# ---------------- GENERATION ----------------

data = []
seen = set()

while len(data) < target_size:

    dept = random.choice(departments)
    sent = random.choices(sentiments, weights=[0.7, 0.3])[0]

    base = get_base(dept, sent)
    template = random.choice(templates)

    text = template.format(base=base)

    text = add_extra(text)
    text = add_case_noise(text)
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

df = df.reset_index(drop=True)
df.insert(0, "id", df.index)

# ---------------- SAVE ----------------

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
DATA_PATH = os.path.join(BASE_DIR, "data", "reviews_noisy.csv")

os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

df.to_csv(DATA_PATH, index=False, encoding="utf-8-sig")

print(f"✅ Generated {len(df)} NOISY reviews (SOFT NOISE)")
print("📁 Saved in:", DATA_PATH)