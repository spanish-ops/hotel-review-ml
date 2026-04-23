import random
import pandas as pd

departments = ["Housekeeping", "Reception", "F&B"]

sentiment_labels = ["pos", "neg"]

# ---------------- HOUSEKEEPING ----------------
hk_pos = [
    "Camera molto pulita e ordinata, biancheria profumata",
    "Pulizia eccellente, bagno impeccabile e ambiente igienizzato",
    "Stanza sempre pulita ogni giorno, servizio rifacimento letto preciso",
    "Camera luminosa, pavimenti puliti e nessuna polvere",
    "Ottima igiene generale della stanza e cambio asciugamani regolare"
]

hk_neg = [
    "Camera sporca, polvere visibile sui mobili",
    "Bagno non pulito e cattivo odore",
    "Servizio di pulizia superficiale e poco accurato",
    "Letto non rifatto correttamente e lenzuola macchiate",
    "Scarsa igiene nella stanza e pavimenti sporchi"
]

# ---------------- RECEPTION ----------------
reception_pos = [
    "Check-in veloce e personale molto gentile",
    "Reception disponibile e professionale, risposte rapide",
    "Accoglienza ottima, personale cortese e sorridente",
    "Procedure di check-in e check-out semplici e veloci",
    "Staff della reception molto efficiente e disponibile 24h"
]

reception_neg = [
    "Attesa molto lunga al check-in",
    "Personale poco cordiale alla reception",
    "Procedure lente e disorganizzate",
    "Scarsa comunicazione alla reception",
    "Check-in confuso e gestione pessima delle richieste"
]

# ---------------- F&B ----------------
fb_pos = [
    "Colazione ricca e varia con prodotti freschi",
    "Ristorante eccellente con piatti gustosi",
    "Buffet abbondante e qualità del cibo ottima",
    "Servizio colazione rapido e cibo di qualità",
    "Ampia scelta di cibo dolce e salato al buffet"
]

fb_neg = [
    "Colazione scarsa e poco varia",
    "Cibo freddo e qualità bassa al ristorante",
    "Servizio lento durante i pasti",
    "Buffet povero e poca scelta",
    "Qualità del cibo sotto le aspettative"
]

def generate(dept, sentiment):
    if dept == "Housekeeping":
        return random.choice(hk_pos if sentiment == "pos" else hk_neg)
    elif dept == "Reception":
        return random.choice(reception_pos if sentiment == "pos" else reception_neg)
    else:
        return random.choice(fb_pos if sentiment == "pos" else fb_neg)


data = []

for i in range(250):  # più dati = meglio ML
    dept = random.choice(departments)
    sent = random.choices(sentiment_labels, weights=[0.7, 0.3])[0]

    text = generate(dept, sent)

    data.append({
        "id": i,
        "text": text,
        "department": dept,
        "sentiment": sent
    })

df = pd.DataFrame(data)
df.to_csv("data/reviews_clean.csv", index=False)

print("Dataset clean migliorato creato!")