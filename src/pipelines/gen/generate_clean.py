import os
import pandas as pd
import random

# ---------------- CONFIG ----------------
departments = ["Housekeeping", "Reception", "F&B"]
sentiment_labels = ["pos", "neg"]

# ---------------- HOUSEKEEPING ----------------
hk_pos_base = [
    "camera pulita e ordinata",
    "bagno impeccabile",
    "stanza sempre pulita",
    "pavimenti senza polvere",
    "ottima igiene generale",
    "camera ben sistemata",
    "ambiente profumato e pulito",
    "pulizia giornaliera accurata",
    "lenzuola fresche e pulite",
    "stanza accogliente e pulita",
    "pulizia delle camere eccellente",
    "bagno ben igienizzato"
]

hk_neg_base = [
    "camera sporca",
    "bagno non pulito",
    "pulizia superficiale",
    "lenzuola macchiate",
    "scarsa igiene",
    "presenza di polvere sui mobili",
    "cattivo odore nella stanza",
    "pulizia insufficiente",
    "bagno sporco e trascurato",
    "servizio di pulizia poco attento",
    "stanza poco curata",
    "igiene generale scarsa"
]

# ---------------- RECEPTION ----------------
rec_pos_base = [
    "check-in veloce",
    "personale gentile",
    "accoglienza ottima",
    "procedure semplici",
    "staff disponibile",
    "personale molto cordiale",
    "servizio reception efficiente",
    "assistenza rapida e professionale",
    "staff sempre presente",
    "accoglienza calorosa",
    "personale disponibile a ogni richiesta",
    "gestione clienti eccellente"
]

rec_neg_base = [
    "attesa lunga al check-in",
    "personale scortese",
    "procedure lente",
    "comunicazione scarsa",
    "gestione pessima",
    "personale poco disponibile",
    "check-in disorganizzato",
    "servizio reception inefficiente",
    "risposte lente alle richieste",
    "accoglienza fredda",
    "mancanza di assistenza",
    "staff poco professionale"
]

# ---------------- F&B ----------------
fb_pos_base = [
    "colazione ricca",
    "cibo ottimo",
    "buffet abbondante",
    "servizio rapido",
    "ampia scelta",
    "prodotti freschi e di qualità",
    "colazione varia e gustosa",
    "piatti ben preparati",
    "ottima qualità del cibo",
    "servizio ristorante efficiente",
    "bevande e cibo ben forniti",
    "esperienza culinaria piacevole"
]

fb_neg_base = [
    "colazione scarsa",
    "cibo freddo",
    "servizio lento",
    "buffet povero",
    "qualità bassa",
    "poca varietà di cibo",
    "piatti poco curati",
    "servizio ristorante lento",
    "cibo di scarsa qualità",
    "colazione deludente",
    "prodotti non freschi",
    "esperienza culinaria negativa"
]

# ---------------- VARIAZIONI ----------------
intensifiers_pos = ["", "molto", "davvero", "estremamente"]
intensifiers_neg = ["", "poco", "per niente", "decisamente"]

extra_context = [
    "",
    " durante il soggiorno",
    " per tutta la permanenza",
    " nel complesso",
    " rispetto alle aspettative"
]

# ---------------- GENERATORE FRASE ----------------
def build_sentence(base_list, sentiment):
    base = random.choice(base_list)

    if sentiment == "pos":
        intensity = random.choice(intensifiers_pos)
    else:
        intensity = random.choice(intensifiers_neg)

    context = random.choice(extra_context)

    sentence = f"{base} {intensity}{context}".strip()
    return sentence


# ---------------- GENERATORE PRINCIPALE ----------------
def generate_review(dept, sentiment):
    if dept == "Housekeeping":
        base_list = hk_pos_base if sentiment == "pos" else hk_neg_base
    elif dept == "Reception":
        base_list = rec_pos_base if sentiment == "pos" else rec_neg_base
    else:
        base_list = fb_pos_base if sentiment == "pos" else fb_neg_base

    return build_sentence(base_list, sentiment)


# ---------------- CREAZIONE DATASET ----------------
data = []
seen_texts = set()

i = 0
target_size = 300  # puoi aumentare

while len(data) < target_size:
    dept = random.choice(departments)
    sent = random.choices(sentiment_labels, weights=[0.7, 0.3])[0]

    text = generate_review(dept, sent)

    # 🔥 evita duplicati
    if text in seen_texts:
        continue

    seen_texts.add(text)

    data.append({
        "id": i,
        "text": text,
        "department": dept,
        "sentiment": sent
    })

    i += 1



# 📍 risalgo fino alla root del progetto
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))

# 📍 punto alla cartella data
DATA_PATH = os.path.join(BASE_DIR, "data", "reviews_clean.csv")

# creo la cartella se non esiste
os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

# --- esempio dataset ---
data = [{"id": i, "text": "test", "department": "F&B", "sentiment": "pos"} for i in range(10)]

df = pd.DataFrame(data)

df.to_csv(DATA_PATH, index=False, encoding="utf-8-sig")

print("✅ Salvato in:", DATA_PATH)