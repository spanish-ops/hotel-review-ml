import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from preprocessing import preprocess_dataframe


# 1. Caricamento dataset
df = pd.read_csv("data/reviews.csv")

# 2. Preprocessing
df = preprocess_dataframe(df)

# 3. Feature e target
X = df["text"]
y_dept = df["department"]
y_sent = df["sentiment"]


# 4. Split dati (stesso split per entrambi)
X_train, X_test, y_train_dept, y_test_dept = train_test_split(
    X, y_dept, test_size=0.2, random_state=42
)

_, _, y_train_sent, y_test_sent = train_test_split(
    X, y_sent, test_size=0.2, random_state=42
)


# 5. Modello reparto
dept_model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2))),
    ("clf", LogisticRegression(max_iter=1000))
])

dept_model.fit(X_train, y_train_dept)


# 6. Modello sentiment
sent_model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2))),
    ("clf", LogisticRegression(max_iter=1000))
])

sent_model.fit(X_train, y_train_sent)


# 7. Salvataggio modelli
joblib.dump(dept_model, "models/dept_model.pkl")
joblib.dump(sent_model, "models/sent_model.pkl")

print("Modelli addestrati e salvati con successo.")