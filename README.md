 Hotel Review ML Project

Progetto di Machine Learning per la classificazione automatica di recensioni di hotel in base a:
- Sentiment (positivo/negativo)
- Dipartimento (Housekeeping, Reception, F&B)

 Obiettivo

L'obiettivo del progetto è costruire un sistema di Natural Language Processing in grado di:
- analizzare recensioni testuali di hotel
- classificare il sentiment
- identificare il reparto coinvolto

 Dataset

Il dataset è composto da tre tipologie di dati:
- recensioni reali (raccolte manualmente)
- recensioni generate "clean"
- recensioni generate con rumore (noisy)

Totale dataset: ~550+ recensioni

 Struttura del progetto

hotel-review-ml/
│
├── data/ # dataset CSV
├── src/ # codice principale
├── notebooks/ # analisi ed esperimenti
├── outputs/ # risultati e grafici
├── models/ # modelli salvati
└── README.md

 Pipeline del progetto

1. Creazione dataset (real + synthetic)
2. Preprocessing del testo
3. Vectorizzazione (TF-IDF)
4. Training modelli ML
5. Valutazione e confronto modelli

 Modelli utilizzati

- Logistic Regression
- Multinomial Naive Bayes
- Support Vector Machine (SVM)


 Metriche

Le performance vengono valutate tramite:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

 Come eseguire il progetto

```bash id="r2"
# creare ambiente virtuale
python -m venv venv
source venv/Scripts/activate  # Windows

# installare dipendenze
pip install -r requirements.txt

# training
python src/evaluate_models.py