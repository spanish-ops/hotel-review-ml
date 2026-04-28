import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def train_models(df, models_path):
    X = df['text']
    y_dep = df['department']
    y_sent = df['sentiment']

    vectorizer = TfidfVectorizer(max_features=5000)

    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_dep_train, y_dep_test, y_sent_train, y_sent_test = train_test_split(
        X_vec, y_dep, y_sent, test_size=0.2, random_state=42
    )

    # Modello reparto
    dep_model = LogisticRegression(max_iter=1000)
    dep_model.fit(X_train, y_dep_train)

    # Modello sentiment
    sent_model = LogisticRegression(max_iter=1000)
    sent_model.fit(X_train, y_sent_train)

    # Salvataggio
    os.makedirs(models_path, exist_ok=True)

    joblib.dump(dep_model, os.path.join(models_path, "department_model.pkl"))
    joblib.dump(sent_model, os.path.join(models_path, "sentiment_model.pkl"))
    joblib.dump(vectorizer, os.path.join(models_path, "vectorizer.pkl"))

    return X_test, y_dep_test, y_sent_test, dep_model, sent_model, vectorizer