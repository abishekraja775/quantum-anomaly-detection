# Quantum Anomaly Detection (Quantum Machine Learning)

This project implements an **unsupervised quantum anomaly detection system**
using a **hybrid quantum–classical variational model**.

The system is evaluated on a **real-world credit card fraud dataset** and
compared against a classical Isolation Forest baseline.

---

## 🔹 Key Features
- Variational Quantum Circuit (Qiskit)
- Unsupervised anomaly detection
- Real-world financial dataset
- ROC–AUC based evaluation
- Classical baseline comparison

---

## 🔹 Project Structure

quantum_anomaly_detection/
├── run_experiment.py
├── quantum_model.py
├── data_loader.py
├── preprocess.py
├── requirements.txt
└── README.md

---


---

## 🔹 Dataset

The **Credit Card Fraud Dataset** is not included in this repository due to
licensing restrictions.

Download it from:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

After downloading, place `creditcard.csv` in the project root directory.

---

## 🔹 How to Run

Install dependencies:

```bash
pip install -r requirements.txt

Run the experiment:

python run_experiment.py **

---

## 🔹 Output

Running the experiment generates the following outputs:

- Scatter plot of **quantum anomaly scores** for normal vs fraudulent samples  
- ROC curve showing the **quantum model performance (AUC)**  
- ROC comparison between **quantum model and Isolation Forest baseline**

These plots appear sequentially during execution.

---

## 🔹 Disclaimer

This project uses **quantum circuit simulation**, which is standard practice
in current quantum machine learning research due to limited access to
large-scale quantum hardware.

No claim of quantum advantage is made.  
The goal is to explore the feasibility of **quantum representations for
unsupervised anomaly detection**.

---

## 🔹 Acknowledgement

This project was developed as part of a **Quantum Computing Hackathon**
focused on Quantum Intelligence and Quantum Machine Learning.



