from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os
import pandas as pd


def evaluate(dep_model, sent_model, X_test, y_dep_test, y_sent_test, output_path):

    # Predizioni
    dep_pred = dep_model.predict(X_test)
    sent_pred = sent_model.predict(X_test)

    # ======================
    # METRICHE
    # ======================
    print("=== REPARTO ===")
    dep_acc = accuracy_score(y_dep_test, dep_pred)
    dep_f1 = f1_score(y_dep_test, dep_pred, average='macro')

    print("Accuracy:", dep_acc)
    print("F1:", dep_f1)
    print(classification_report(y_dep_test, dep_pred))

    print("\n=== SENTIMENT ===")
    sent_acc = accuracy_score(y_sent_test, sent_pred)
    sent_f1 = f1_score(y_sent_test, sent_pred, average='macro')

    print("Accuracy:", sent_acc)
    print("F1:", sent_f1)
    print(classification_report(y_sent_test, sent_pred))

    # ======================
    # CARTELLE OUTPUT
    # ======================
    os.makedirs(output_path, exist_ok=True)
    figures_path = os.path.join(output_path, "figures")
    os.makedirs(figures_path, exist_ok=True)

    # ======================
    # SALVATAGGIO METRICHE
    # ======================
    with open(os.path.join(output_path, "metrics.txt"), "w") as f:
        f.write("=== DEPARTMENT ===\n")
        f.write(f"Accuracy: {dep_acc}\n")
        f.write(f"F1: {dep_f1}\n\n")

        f.write("=== SENTIMENT ===\n")
        f.write(f"Accuracy: {sent_acc}\n")
        f.write(f"F1: {sent_f1}\n")

    # ======================
    # CONFUSION MATRIX - DEPARTMENT
    # ======================
    cm_dep = confusion_matrix(y_dep_test, dep_pred)

    plt.figure()
    plt.imshow(cm_dep)
    plt.title("Confusion Matrix - Department")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(figures_path, "cm_department.png"))
    plt.close()

    # ======================
    # CONFUSION MATRIX - SENTIMENT
    # ======================
    cm_sent = confusion_matrix(y_sent_test, sent_pred)

    plt.figure()
    plt.imshow(cm_sent)
    plt.title("Confusion Matrix - Sentiment")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(figures_path, "cm_sentiment.png"))
    plt.close()

    # ======================
    # BAR CHART - DEPARTMENT
    # ======================
    df_dep = pd.DataFrame({"department": y_dep_test})
    counts_dep = df_dep["department"].value_counts()

    plt.figure()
    counts_dep.plot(kind='bar')
    plt.title("Distribuzione classi (Department)")
    plt.xlabel("Classe")
    plt.ylabel("Numero esempi")
    plt.savefig(os.path.join(figures_path, "class_distribution.png"))
    plt.close()

    # ======================
    # BAR CHART - SENTIMENT
    # ======================
    df_sent = pd.DataFrame({"sentiment": y_sent_test})
    counts_sent = df_sent["sentiment"].value_counts()

    plt.figure()
    counts_sent.plot(kind='bar')
    plt.title("Distribuzione classi (Sentiment)")
    plt.xlabel("Classe")
    plt.ylabel("Numero esempi")
    plt.savefig(os.path.join(figures_path, "sentiment_distribution.png"))
    plt.close()

    print("\n✔ Tutti i risultati salvati in:", output_path)