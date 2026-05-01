# 🏨 Hotel Review ML Project

Progetto di Machine Learning per la classificazione automatica di recensioni di hotel basato su:

- 📌 Sentiment analysis (positivo / negativo)
- 📌 Classificazione del reparto (Housekeeping, Reception, Food & Beverage)

---

## 🎯 Obiettivo

L’obiettivo del progetto è sviluppare un sistema di Natural Language Processing in grado di:

- analizzare recensioni testuali non strutturate
- identificare il reparto di riferimento della recensione
- classificare il sentiment espresso dal cliente
- fornire una soluzione applicativa utilizzabile tramite dashboard

Il sistema è stato progettato per simulare un contesto reale di analisi dei feedback dei clienti in ambito alberghiero.

---

## 📊 Dataset

Il dataset è composto da circa 500+ recensioni e include:

- recensioni sintetiche pulite
- recensioni sintetiche con rumore (typos, maiuscole, ambiguità)
- recensioni reali raccolte online

Ogni record contiene:
- `text` → recensione
- `department` → reparto
- `sentiment` → polarità

Il dataset finale è stato bilanciato e mescolato per migliorare la generalizzazione del modello.

---

## 🧠 Modelli utilizzati

- Logistic Regression (reparto)
- Logistic Regression (sentiment)
- TF-IDF Vectorizer per la rappresentazione testuale

Modelli salvati nella cartella:

models/
├── department_model.pkl
├── sentiment_model.pkl
└── vectorizer.pkl

---

## 📁 Struttura del progetto
hotel-review-ml/
│
├── data/
│ └── reviews_dataset.csv
│
├── src/
│ ├── main.py
│ ├── pipelines/
│ └── ml/
│ ├── preprocessing.py
│ ├── train_models.py
│ ├── evaluate_models.py
│ └── explainability.py
│
├── models/
├── outputs/
│ ├── figures/
│
├── app/
│ └── dashboard.py
│
└── README.md


---

## 🚀 Come eseguire il progetto

### 1️⃣ Creazione ambiente virtuale

```bash
python -m venv venv
Attivazione:

Windows:

venv\Scripts\activate

2️⃣ Installazione dipendenze
pip install -r requirements.txt


3️⃣ Esecuzione pipeline ML (training + evaluation)

Dalla root del progetto:
python -m src.main

Questo comando esegue:

caricamento dataset
preprocessing
training modelli
valutazione
salvataggio output in outputs/

📊 Output generati

Dopo l’esecuzione:

metriche (accuracy, F1-score)
confusion matrix salvate in:
outputs/figures/

🖥️ Avvio Dashboard


La dashboard permette di:

inserire una singola recensione
oppure caricare un file CSV
ottenere:
reparto predetto
sentiment predetto
probabilità associate


▶ Avvio:
streamlit run app/dashboard.py

📤 Export risultati

Quando si carica un file CSV dalla dashboard:

il sistema genera automaticamente un file di output
il file viene salvato con timestamp

Percorso:

outputs/predictions/