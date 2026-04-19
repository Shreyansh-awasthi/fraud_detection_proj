# 🛡️ Fraud Detection System
### by Shreyansh Awasthi

> End-to-end machine learning pipeline for real-time financial fraud detection — trained on 6.3M transactions with extreme class imbalance (0.13% fraud rate), deployed as an interactive Streamlit web application.

---

## 📌 Project Overview

Financial fraud detection is one of the hardest real-world ML problems — not because the algorithms are complex, but because the data is brutally imbalanced. Out of 6.3 million transactions, only **8,190 are fraud (0.13%)**. A naive model that predicts "not fraud" every time would achieve 99.87% accuracy while catching zero fraud cases.

This project solves that problem properly.

---

## 🧠 Key Design Decisions

### Why only TRANSFER and CASH_OUT?
The dataset contains 6 transaction types. After analysis, **PAYMENT and CASH_IN had zero fraud cases** across 4M+ rows. Keeping them would have:
- Artificially inflated accuracy
- Taught the model "these types = never fraud" — not intelligence, just a shortcut
- Created distributional contamination

**Smart data filtering is more important than more data.**

### Why `scale_pos_weight` instead of SMOTE?
With 0.13% fraud rate, the class weight ratio is **~768:1**.

| | SMOTE | scale_pos_weight |
|---|---|---|
| Speed | Slow (generates synthetic rows) | Fast |
| Overfitting risk | Higher | Lower |
| Production safety | Needs fit on train only | No leakage risk |

At 6.3M rows there's already enough signal — `scale_pos_weight` reweights mathematically without generating fake data.

---

## 📊 Model Comparison

| Model | Accuracy | Notes |
|---|---|---|
| Logistic Regression | 93.00% | Baseline — poor on imbalanced data |
| Random Forest | 99.92% | Strong but slower inference |
| CalibratedClassifierCV | 99.03% | Probability calibration issues |
| **XGBoost** ✅ | **99.63%** | **Best — chosen for deployment** |

> ⚠️ Accuracy alone is misleading here. The real metrics that matter are **Precision and Recall**.

### Final Model — XGBoost
| Metric | Score |
|---|---|
| Precision | **87%** |
| Recall | **90%** |
| F1 Score | **88%** |
| Threshold | **0.95** |
| Unique probability values | **533,454 / 554,082** |

**90% recall** means catching 9 out of every 10 fraud cases.
**87% precision** means 87% of fraud alerts are real fraud.

---

## 🚀 Running the App

### Prerequisites
```bash
pip install -r requirements.txt
```

### Launch
```bash
streamlit run app.py
```

The app will automatically download the model from Google Drive on first run.

---

## 📱 Streamlit App Features

- **Real-time prediction** — enter transaction details, click Predict
- **Auto-computed features** — log_amount, balance_diff, is_drained calculated automatically
- **Risk levels** — LOW / MEDIUM / HIGH based on probability score
- **Confidence gap** — how far the prediction is from the threshold
- **Clean UI** — dark theme, no clutter

---

## 📈 Threshold Analysis

At 0.13% fraud rate with 6.3M transactions (~8,190 fraud cases):

| Threshold | Precision | Recall | Fraud Caught | Missed Fraud |
|---|---|---|---|---|
| 0.75 | ~76% | ~95% | 7,781 | 409 |
| 0.85 | ~83% | ~92% | 7,535 | 655 |
| **0.95** ✅ | **87%** | **90%** | **7,371** | **819** |
| 0.99 | ~92% | ~76% | 6,224 | 1,966 |

Threshold 0.95 was chosen as the best balance between catching fraud and minimizing false alerts.

---

## 💡 What I Learned

- **Accuracy is a lie** on imbalanced datasets — always use Precision-Recall
- **Data filtering matters** more than algorithm choice sometimes
- **Calibration can hurt** — CalibratedClassifierCV collapsed probabilities for minority transaction types
- **scale_pos_weight** is cleaner and more stable than SMOTE at large scale
- **Threshold tuning** is a business decision, not just a technical one

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.11-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-latest-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-latest-red)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-latest-green)
![Pandas](https://img.shields.io/badge/Pandas-latest-lightblue)

---

## 👤 Author

**Shreyansh Awasthi**

---

> *"The best fraud detection model isn't the most complex one — it's the one that understands the problem."*
>
> **Live Demo:**[https://frauddetectionproj-qf89vnabq6spjgqmdunmw7.streamlit.app/]
