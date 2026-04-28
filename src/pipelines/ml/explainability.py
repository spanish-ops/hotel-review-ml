from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os

def evaluate(dep_model, sent_model, X_test, y_dep_test, y_sent_test, output_path):

    dep_pred = dep_model.predict(X_test)
    sent_pred = sent_model.predict(X_test)

    print("=== REPARTO ===")
    print("Accuracy:", accuracy_score(y_dep_test, dep_pred))
    print("F1:", f1_score(y_dep_test, dep_pred, average='macro'))
    print(classification_report(y_dep_test, dep_pred))

    print("\n=== SENTIMENT ===")
    print("Accuracy:", accuracy_score(y_sent_test, sent_pred))
    print("F1:", f1_score(y_sent_test, sent_pred, average='macro'))
    print(classification_report(y_sent_test, sent_pred))

    os.makedirs(output_path, exist_ok=True)

    # Confusion Matrix reparto
    cm_dep = confusion_matrix(y_dep_test, dep_pred)
    plt.figure()
    plt.imshow(cm_dep)
    plt.title("Confusion Matrix - Department")
    plt.colorbar()
    plt.savefig(os.path.join(output_path, "cm_department.png"))

    # Confusion Matrix sentiment
    cm_sent = confusion_matrix(y_sent_test, sent_pred)
    plt.figure()
    plt.imshow(cm_sent)
    plt.title("Confusion Matrix - Sentiment")
    plt.colorbar()
    plt.savefig(os.path.join(output_path, "cm_sentiment.png"))