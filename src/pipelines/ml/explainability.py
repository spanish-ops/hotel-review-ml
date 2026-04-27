import numpy as np
import pandas as pd
import joblib


# Carica modello già addestrato
model = joblib.load("models/dept_model.pkl")


def explain_prediction(text):
    """
    Mostra le parole più influenti per una predizione
    """

    # estrai TF-IDF e classificatore
    tfidf = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]

    # trasformazione testo
    X = tfidf.transform([text])

    # predizione
    prediction = model.predict([text])[0]

    # recupero parole
    feature_names = tfidf.get_feature_names_out()

    # pesi del modello (coeff logistic regression)
    coef = clf.coef_[0]

    # associa parole → peso
    word_weights = list(zip(feature_names, coef))

    # ordina per importanza
    top_positive = sorted(word_weights, key=lambda x: x[1], reverse=True)[:10]
    top_negative = sorted(word_weights, key=lambda x: x[1])[:10]

    return {
        "text": text,
        "prediction": prediction,
        "top_positive_words": top_positive,
        "top_negative_words": top_negative
    }


# esempio di test
if __name__ == "__main__":
    result = explain_prediction("personale scortese e camera sporca")
    print(result)