import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import IsolationForest

from data_loader import load_creditcard_data
from preprocess import preprocess_data
from quantum_model import quantum_circuit, get_expectation


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "creditcard.csv")


def loss_function(params, data):
    losses = []
    for x in data:
        qc = quantum_circuit(x, params)
        losses.append(get_expectation(qc) ** 2)
    return np.mean(losses)


normal_df, anomaly_df = load_creditcard_data(csv_path)

normal_train = normal_df.sample(200, random_state=42).values
normal_test = normal_df.sample(50, random_state=1).values
anomaly_test = anomaly_df.sample(50, random_state=42).values

test_data = np.vstack([normal_test, anomaly_test])
test_labels = np.array([0]*len(normal_test) + [1]*len(anomaly_test))

normal_q, test_q = preprocess_data(normal_train, test_data)


init_params = np.random.rand(2)
result = minimize(loss_function, init_params, args=(normal_q,), method="COBYLA")
trained_params = result.x

print("Trained quantum parameters:", trained_params)


scores = []
for x in test_q:
    qc = quantum_circuit(x, trained_params)
    scores.append(abs(get_expectation(qc)))


plt.figure(figsize=(8, 4))
plt.scatter(range(len(scores)), scores, c=test_labels, cmap="coolwarm")
plt.xlabel("Sample Index")
plt.ylabel("Quantum Anomaly Score")
plt.title("Quantum Anomaly Scores on Credit Card Data")
plt.colorbar(label="0 = Normal, 1 = Fraud")
plt.show()


fpr, tpr, _ = roc_curve(test_labels, scores)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"Quantum ROC (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Quantum Anomaly Detector")
plt.legend()
plt.show()

print("Quantum Model AUC:", roc_auc)


iso = IsolationForest(contamination=0.05, random_state=42)
iso.fit(normal_q)

iso_scores = -iso.decision_function(test_q)

fpr_c, tpr_c, _ = roc_curve(test_labels, iso_scores)
auc_c = auc(fpr_c, tpr_c)

plt.figure()
plt.plot(fpr, tpr, label=f"Quantum (AUC = {roc_auc:.3f})")
plt.plot(fpr_c, tpr_c, label=f"Isolation Forest (AUC = {auc_c:.3f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Quantum vs Classical Anomaly Detection")
plt.legend()
plt.show()

print("Classical Isolation Forest AUC:", auc_c)

