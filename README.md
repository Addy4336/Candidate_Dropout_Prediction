# 🎯 Candidate Dropout Prediction

A machine learning project to predict whether a candidate will drop out before the final interview — helping recruiters reduce pipeline friction and improve hiring outcomes.

---

## 📌 Problem Statement
Final interview "no-shows" waste recruiter time and slow down hiring. This project aims to predict dropout risk early using candidate profile & behavior features.

---

## 🧠 Highlights
- Built with **Logistic Regression** (interpretable, fast)
- Covers all modeling assumptions: multicollinearity, log-odds, class imbalance
- 100% **recall** for dropouts via threshold tuning
- Fully deployed with **Streamlit**
- Accepts both single profile inputs & **CSV batch uploads**
- Dashboard with **feature importance** & **ROC curve**

---

## 🧪 Model Performance
| Metric         | Value |
|----------------|--------|
| **Recall (1)** | 100%   |
| Precision (1)  | 14%    |
| AUC            | 0.70   |

> Focused on catching all likely dropouts for proactive recruiter action.

---

## 🚀 Try the App
Clone this repo and run:
```bash
streamlit run app.py
