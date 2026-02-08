# Quantum Anomaly Detection (Quantum ML)

This project implements an **unsupervised quantum anomaly detection system**
using a **hybrid quantum–classical variational model**.

The model is evaluated on a **real-world credit card fraud dataset** and
compared against a classical Isolation Forest baseline.

---

## 🔹 Key Features
- Variational Quantum Circuit (Qiskit)
- Unsupervised anomaly detection
- Real-world financial dataset
- ROC–AUC evaluation
- Classical baseline comparison

---

## 🔹 Project Structure

quantum_anomaly_detection/
│
├── run_experiment.py
├── quantum_model.py
├── data_loader.py
├── preprocess.py
├── requirements.txt
└── README.md


---

## 🔹 Dataset
The **Credit Card Fraud Dataset** is not included due to licensing.

Download from:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place `creditcard.csv` in the project root before running.

---

## 🔹 How to Run

```bash
pip install -r requirements.txt
python run_experiment.py

## 🔹 Output

Quantum anomaly score scatter plot

ROC curve (Quantum model)

ROC comparison with Isolation Forest

## 🔹 Disclaimer

This project uses quantum simulation, which is standard practice in
current quantum machine learning research.
No claim of quantum advantage is made.


