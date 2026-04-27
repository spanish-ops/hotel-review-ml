import re

def clean_text(text):
    text = text.lower()  # minuscolo
    text = re.sub(r"[^\w\s]", " ", text)  # rimuove punteggiatura
    text = re.sub(r"\s+", " ", text).strip()  # spazi multipli
    return text


def preprocess_dataframe(df):
    df = df.copy()
    df["text"] = df["text"].apply(clean_text)
    return df