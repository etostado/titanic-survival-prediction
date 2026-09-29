# 🚢 Titanic Survival Prediction

An end-to-end machine learning project that predicts whether a Titanic passenger survived, built with Python, pandas and scikit-learn.

**Best model:** Logistic Regression, **83.8% accuracy** on unseen data, with no overfitting (83.4% train vs 83.8% test).

---

## 📌 Project overview

| | |
|---|---|
| **Problem** | Binary classification (survived = 1, did not survive = 0) |
| **Data** | 891 passengers, 12 original columns ([Kaggle Titanic dataset](https://www.kaggle.com/c/titanic)) |
| **Tools** | Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn |
| **Models** | Logistic Regression, Decision Tree, Random Forest |

## 🔍 Key EDA findings

| Factor | Survival rate |
|---|---|
| Overall | 38% |
| Women vs men | **74%** vs **19%** |
| 1st vs 3rd class | **63%** vs **24%** |
| Children (0–10) | 59%, the highest of any age group |
| Had a recorded cabin vs none | 67% vs 30% |
| Title *Mrs* vs *Mr* | 79% vs 16% |

## 🛠️ What I did

1. **Data cleaning:** filled `Embarked` with the mode, filled `Age` with the median of each class + sex group, and turned the 77%-empty `Cabin` column into a `HasCabin` flag.
2. **Feature engineering:** extracted passenger **Title** from names (Mr, Mrs, Miss, Master, Rare) and created **FamilySize** and **IsAlone**.
3. **Encoding:** mapped `Sex` to 0/1 and one-hot encoded `Embarked` and `Title`.
4. **Train/test split:** 80/20, stratified to keep the same survival ratio in both sets.
5. **Modeling & evaluation:** accuracy, confusion matrix, precision/recall.
6. **Overfitting diagnosis:** compared train vs test scores and tuned `max_depth`.
7. **What-if predictions** on made-up passengers, plus **explainability** through model coefficients.

## 📊 Results

| Model | Train | Test | Notes |
|---|---|---|---|
| **Logistic Regression** | 83.4% | **83.8%** | 🏆 Best, no overfitting |
| Random Forest (max_depth=5) | 85.0% | 82.1% | |
| Decision Tree (max_depth=5) | 86.9% | 81.0% | |
| Random Forest (default) | 98.7% | 79.9% | ⚠️ Overfitting |
| Decision Tree (default) | 98.7% | 77.7% | ⚠️ Overfitting |

**Confusion matrix (Logistic Regression, 179 test passengers):** 150 correct predictions. Recall is 88% for non-survivors and 77% for survivors.

## 💡 Takeaways

- **Sex, title and ticket class** were the strongest predictors, reflecting the "women and children first" rule and the class divide on board.
- **More complex isn't always better:** on a small dataset, a simple Logistic Regression beat Random Forest.
- **A 99% training score is a red flag, not a win.** Unconstrained trees memorized passengers; limiting depth improved performance on new data.
- **What-if test:** the same passenger (woman, 39, traveling alone) goes from **45% → 83%** survival probability just by upgrading from a 3rd- to a 1st-class ticket.

## ▶️ How to run

** Locally**
```bash
git clone https://github.com/<your-username>/titanic-survival-prediction.git
cd titanic-survival-prediction
pip install -r requirements.txt
jupyter notebook titanic_survival_prediction.ipynb
```

## 🚀 Next steps

- Cross-validation and `GridSearchCV` for more robust tuning
- Serve the model through a Flask API or a Gradio web app
- Try gradient boosting (XGBoost / LightGBM)

---

**Author:** Edith Tostado Ortiz · Built as part of my journey through the IBM Generative AI Engineering Professional Certificate.
