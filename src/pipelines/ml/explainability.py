import numpy as np

def show_top_features(model, vectorizer, n=10):
    feature_names = vectorizer.get_feature_names_out()

    for i, class_label in enumerate(model.classes_):
        top = np.argsort(model.coef_[i])[-n:]
        print(f"\nClasse: {class_label}")
        print([feature_names[j] for j in top])