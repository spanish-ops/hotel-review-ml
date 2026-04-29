import os
import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# =========================
# PATH
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

DEP_MODEL_PATH = os.path.join(MODELS_DIR, "department_model.pkl")
SENT_MODEL_PATH = os.path.join(MODELS_DIR, "sentiment_model.pkl")
VECT_PATH = os.path.join(MODELS_DIR, "vectorizer.pkl")

# =========================
# LOAD MODELS
# =========================
dep_model = joblib.load(DEP_MODEL_PATH)
sent_model = joblib.load(SENT_MODEL_PATH)
vectorizer = joblib.load(VECT_PATH)

# =========================
# FUNZIONE PREDIZIONE
# =========================
def predict(text):
    X = vectorizer.transform([text])

    dep = dep_model.predict(X)[0]
    sent = sent_model.predict(X)[0]

    dep_proba = dep_model.predict_proba(X).max()
    sent_proba = sent_model.predict_proba(X).max()

    return dep, dep_proba, sent, sent_proba


# =========================
# UI
# =========================
st.title("🏨 Hotel Review AI Dashboard")

menu = st.sidebar.selectbox("Menu", ["Single prediction", "Batch CSV"])

# =========================
# 1. SINGLE PREDICTION
# =========================
if menu == "Single prediction":

    text = st.text_area("Inserisci recensione")

    if st.button("Analizza"):

        if text.strip() == "":
            st.warning("Inserisci testo")
        else:
            dep, dep_p, sent, sent_p = predict(text)

            st.subheader("📌 Risultato")

            st.write("🏷 Reparto:", dep)
            st.write("📊 Confidenza reparto:", round(dep_p, 2))

            st.write("😊 Sentiment:", sent)
            st.write("📊 Confidenza sentiment:", round(sent_p, 2))

# =========================
# 2. BATCH CSV
# =========================
elif menu == "Batch CSV":

    file = st.file_uploader("Carica CSV con colonna 'text'", type=["csv"])

    if file is not None:

        df = pd.read_csv(file)

        if "text" not in df.columns:
            st.error("Il CSV deve avere colonna 'text'")
        else:

            results = []

            for t in df["text"]:
                dep, dep_p, sent, sent_p = predict(t)

                results.append({
                    "text": t,
                    "department": dep,
                    "dep_conf": dep_p,
                    "sentiment": sent,
                    "sent_conf": sent_p
                })

            result_df = pd.DataFrame(results)

            st.write(result_df)

            # =========================
            # EXPORT FILE
            # =========================
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(OUTPUT_DIR, f"predictions_{timestamp}.csv")

            os.makedirs(OUTPUT_DIR, exist_ok=True)
            result_df.to_csv(output_path, index=False)

            st.success(f"File salvato: {output_path}")

            st.download_button(
                "Download risultati",
                data=result_df.to_csv(index=False),
                file_name=f"predictions_{timestamp}.csv",
                mime="text/csv"
            )