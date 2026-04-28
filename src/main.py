import os

from src.pipelines.ml.preprocessing import load_and_preprocess
from src.pipelines.ml.train_models import train_models
from src.pipelines.ml.evaluate_models import evaluate
from src.pipelines.ml.explainability import show_top_features

# PATH DINAMICI
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "reviews_dataset.csv")
MODELS_PATH = os.path.join(BASE_DIR, "models")
OUTPUT_PATH = os.path.join(BASE_DIR, "outputs")

def main():
    print("Caricamento e preprocessing...")
    df = load_and_preprocess(DATA_PATH)

    print("Training modelli...")
    X_test, y_dep_test, y_sent_test, dep_model, sent_model, vectorizer = train_models(df, MODELS_PATH)

    print("Valutazione...")
    evaluate(dep_model, sent_model, X_test, y_dep_test, y_sent_test, OUTPUT_PATH)

    print("Explainability (feature importanti)...")
    show_top_features(dep_model, vectorizer)

if __name__ == "__main__":
    main()