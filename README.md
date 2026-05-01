# 🏨 Hotel Review ML Project

Progetto di Machine Learning per la classificazione automatica di recensioni di hotel basato su:
- Sentiment (positivo / negativo)
- Reparto (Housekeeping, Reception, Food & Beverage)

---

## 🎯 Obiettivo

Il progetto ha lo scopo di sviluppare un sistema di Natural Language Processing in grado di:
- analizzare recensioni testuali non strutturate
- classificare il reparto di riferimento
- classificare il sentiment espresso
- fornire un’interfaccia utilizzabile tramite dashboard

---

## 📊 Dataset

Il dataset (~500+ recensioni) include:
- recensioni sintetiche pulite
- recensioni sintetiche con rumore
- recensioni reali

Ogni record contiene:
- text
- department
- sentiment

---

## 🧠 Modelli

- Logistic Regression (department)
- Logistic Regression (sentiment)
- TF-IDF Vectorizer

Modelli salvati in:
- models/department_model.pkl
- models/sentiment_model.pkl
- models/vectorizer.pkl

---

## 📁 Struttura progetto

hotel-review-ml/
│
├── data/
├── src/
│   ├── main.py
│   └── pipelines/ml/
├── models/
├── outputs/
│   └── figures/
├── app/
│   └── dashboard.py

---

## 🚀 Esecuzione progetto

### 1. Ambiente virtuale
python -m venv venv

Windows:
venv\Scripts\activate

---

### 2. Installazione dipendenze
pip install -r requirements.txt

---

### 3. Training modello
python -m src.main

Output:
- metriche
- confusion matrix in outputs/figures/

---

## 🖥️ Dashboard

Funzionalità:
- inserimento recensione singola
- upload CSV
- predizione reparto + sentiment
- probabilità associate

Avvio:
streamlit run app/dashboard.py

---

## 📤 Export risultati

I risultati vengono salvati automaticamente con timestamp in:
outputs/predictions/

---

## 🧪 Tecnologie

Python, Scikit-learn, Pandas, NumPy, TF-IDF, Streamlit, Matplotlib

---

## 📌 Note

- dataset misto (reale + sintetico)
- progetto dimostrativo di NLP
- buone capacità di generalizzazione su testi simili