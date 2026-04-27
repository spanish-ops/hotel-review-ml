import os
import pandas as pd
import random
from itertools import product

# ---------------- CONFIG ----------------
departments = ["Housekeeping", "Reception", "F&B"]
sentiments = ["pos", "neg"]

target_size = 250

# ---------------- BASE PHRASES (ESPANSE) ----------------

hk_pos = [
    "pulita e ordinata",
    "impeccabile",
    "ben mantenuta",
    "igienizzata correttamente",
    "ben curata",
    "fresca e pulita",
    "molto pulita",
    "perfettamente ordinata",
    "in ottime condizioni"
]

hk_neg = [
    "sporca",
    "non pulita",
    "trascurata",
    "con scarsa igiene",
    "piena di polvere",
    "maleodorante",
    "poco curata"
]

rec_pos = [
    "gentile",
    "molto disponibile",
    "professionale",
    "efficiente",
    "cordiale",
    "attento alle richieste",
    "ben organizzato"
]

rec_neg = [
    "scortese",
    "poco disponibile",
    "lento",
    "inefficiente",
    "disorganizzato",
    "poco professionale"
]

fb_pos = [
    "buona",
    "ottima",
    "varia e gustosa",
    "di qualità",
    "ben preparata",
    "apprezzabile",
    "soddisfacente"
]

fb_neg = [
    "scarsa",
    "deludente",
    "fredda",
    "poco varia",
    "di bassa qualità",
    "non soddisfacente"
]

# ---------------- TEMPLATE (ESPANSI) ----------------

templates = {
    "Housekeeping": [
        "La camera è {base}",
        "La camera risulta {base}",
        "Durante il soggiorno la camera è stata {base}",
        "Abbiamo trovato la camera {base}",
        "La camera si presenta {base}",
        "Il livello di pulizia è {base}"
    ],
    "Reception": [
        "Il personale è {base}",
        "Il personale si è dimostrato {base}",
        "Durante il soggiorno il personale è stato {base}",
        "Il servizio di reception è {base}",
        "La reception risulta {base}",
        "L'accoglienza è stata {base}"
    ],
    "F&B": [
        "La colazione è {base}",
        "Il servizio di ristorazione è {base}",
        "Durante il soggiorno il cibo è {base}",
        "Il buffet è {base}",
        "L'offerta gastronomica è {base}",
        "La qualità del cibo è {base}"
    ]
}

# ---------------- MAP ----------------

base_map = {
    ("Housekeeping", "pos"): hk_pos,
    ("Housekeeping", "neg"): hk_neg,
    ("Reception", "pos"): rec_pos,
    ("Reception", "neg"): rec_neg,
    ("F&B", "pos"): fb_pos,
    ("F&B", "neg"): fb_neg
}

# ---------------- GENERAZIONE COMPLETA ----------------

all_combinations = []

for dept, sent in product(departments, sentiments):
    bases = base_map[(dept, sent)]
    for base in bases:
        for template in templates[dept]:
            text = template.format(base=base)

            all_combinations.append({
                "text": text,
                "department": dept,
                "sentiment": sent
            })

# ---------------- SHUFFLE ----------------

random.shuffle(all_combinations)

# ---------------- SELEZIONE 250 UNICHE ----------------

seen = set()
data = []

for item in all_combinations:
    key = item["text"].lower().strip()

    if key in seen:
        continue

    seen.add(key)
    data.append(item)

    if len(data) >= target_size:
        break

# ---------------- DATAFRAME ----------------

for i, row in enumerate(data):
    row["id"] = i

df = pd.DataFrame(data)

# ---------------- SAVE ----------------

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
DATA_PATH = os.path.join(BASE_DIR, "data", "reviews_clean.csv")

os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

df.to_csv(DATA_PATH, index=False, encoding="utf-8-sig")

print(f"✅ Generate {len(df)} UNIQUE reviews (target {target_size})")
print("📁 Saved in:", DATA_PATH)
