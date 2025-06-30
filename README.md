# AI-Based Predictive Maintenance for Cellular Networks

This project uses machine learning models to predict network failures in cellular systems (3G, 4G, LTE, 5G). It applies supervised learning to anticipate potential issues based on signal metrics such as signal strength, latency, and spectrum measurements from multiple sources.

## 🌐 Problem Statement

Modern cellular networks face occasional signal degradation due to hardware limitations, interference, and environmental factors. Predicting these failures in advance allows telecom providers to take proactive measures to reduce downtime and improve Quality of Service (QoS).

## 🧠 Machine Learning Models Used

- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)

## 🗂️ Dataset

- Source: [Kaggle - Cellular Network Analysis Dataset](https://www.kaggle.com/datasets/suraj520/cellular-network-analysis-dataset)
- Augmented by: [5G-Dataset from GitHub](https://github.com/kautik4/5G-dataset)
- Features: 
  - Signal Strength (dBm)
  - BB60C, srsRAN, BladeRFxA9 Measurements
  - Latency (ms)
  - Network Type (3G, 4G, LTE, 5G)
- Target:
  - `Failure`: A binary label (0 = healthy, 1 = failed) based on signal and latency thresholds

## 🛠️ Workflow

1. Load and clean data from multiple sources
2. Engineer `Failure` column based on signal and latency conditions
3. One-hot encode the `Network Type`
4. Train and evaluate ML classifiers
5. Visualize results with confusion matrix and performance metrics

## 📈 Evaluation Metrics

- Accuracy
- Confusion Matrix
- Classification Report (Precision, Recall, F1-score)

## 🔍 Example Result Summary

| Model               | Accuracy |
|--------------------|----------|
| Logistic Regression| 94.7%    |
| Decision Tree      | 92.3%    |
| KNN (k=5)           | 89.6%    |

> (Replace with your actual results after running the full notebook.)

## 🔧 Tech Stack

- Python 3.x
- Jupyter Notebook
- Libraries:
  - `pandas`, `numpy`
  - `scikit-learn`
  - `matplotlib`, `seaborn`

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/cellguard-ai.git
   cd cellguard-ai
