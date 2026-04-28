import re
import pandas as pd

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # rimuove punteggiatura
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_preprocess(path):
    df = pd.read_csv(path)

    df['text'] = df['text'].astype(str).apply(clean_text)

    return df