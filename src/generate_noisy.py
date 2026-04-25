import random
import pandas as pd

def add_noise(text):
    # errori ortografici 
    typos = {
        "camera": "camra",
        "pulita": "pulitA",
        "colazione": "colazzione",
        "servizio": "servizo",
        "personale": "personalee"
    }

    # applica typo casuali
    for k, v in typos.items():
        if random.random() < 0.25:
            text = text.replace(k, v)

    # variazioni di stile
    if random.random() < 0.3:
        text = text + " ma non sempre perfetto"

    if random.random() < 0.2:
        text = text.upper()

    if random.random() < 0.3:
        text = text + "!!!"

    # ambiguità 
    ambiguous_addons = [
        " anche se qualcosa mancava",
        " ma potrebbe migliorare",
        " non tutto era perfetto",
        " esperienza altalenante"
    ]

    if random.random() < 0.25:
        text += random.choice(ambiguous_addons)

    return text


df = pd.read_csv("data/reviews_clean.csv")

noisy_data = []

for i, row in df.iterrows():
    noisy_text = add_noise(row["text"])

    noisy_data.append({
        "id": row["id"],
        "text": noisy_text,
        "department": row["department"],
        "sentiment": row["sentiment"]
    })

df_noisy = pd.DataFrame(noisy_data)
df_noisy.to_csv("data/reviews_noisy.csv", index=False)

print("Dataset noisy migliorato creato!")