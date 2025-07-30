# IIIT-Sonepat-Minor-Project---Predicting-Drug-Consumption-using-ML

# 💊 Drug Consumption Prediction using Machine Learning

A data-driven project aimed at predicting drug consumption patterns using psychological and demographic features. It applies machine learning models to forecast drug use risks and aids in developing public health strategies.

---

## 📌 Problem Statement

Understanding and predicting drug consumption can be a powerful tool for healthcare systems, mental wellness campaigns, and substance abuse prevention. This project leverages user traits like personality scores, age, education, etc., to predict the probability of usage of various drugs, with a particular focus on **Cannabis** consumption.

---

## 🎯 Project Objectives

- Predict whether an individual is a drug user based on key features.
- Use psychological profiling and demographics to detect patterns.
- Evaluate multiple classification algorithms to find the best performer.
- Derive actionable insights for public health stakeholders.

---

## 📂 Dataset Details

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Drug+consumption+%28quantified%29)
- **Name**: Drug Consumption (Quantified)
- **Records**: 1885 individuals
- **Attributes**: 
  - Age, Gender, Education
  - Personality Scores: NEO-FFI (N, E, O, A, C)
  - Impulsiveness, Sensation Seeking
  - Drug Use (binary/multiclass for 19 substances)

---



## 🧰 Tech Stack

| Category      | Tools Used                                       |
|---------------|--------------------------------------------------|
| Language      | Python 3.x                                       |
| IDE/Notebook  | Google Colab / Jupyter Notebook                  |
| Libraries     | Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn |
| Deep Learning | Tensorflow, keras                                |



---

## ⚙️ Installation Guide

Clone the repo and install dependencies:

```bash
git clone https://github.com/yourusername/drug-consumption-prediction.git
cd drug-consumption-prediction
pip install -r requirements.txt
🧪 Exploratory Data Analysis (EDA)



Key Python Commands:

python
Copy
Edit
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("drug_consumption.csv")
df.info()
df.describe()
df.isnull().sum()
🔍 Visualizations
python
Copy
Edit
sns.countplot(data=df, x='Cannabis')
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
sns.pairplot(df[['Age', 'Education', 'Nscore', 'Escore', 'Cannabis']], hue='Cannabis')



✅ Findings:

1. Younger individuals with higher impulsiveness are more likely to consume cannabis.

2. Personality traits like Neuroticism and Sensation Seeking show strong correlation.

3. Education level has mixed influence, but lower education correlates slightly with higher usage.




🧠 Feature Engineering & Preprocessing
python
Copy
Edit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

X = df.drop(columns=['Cannabis'])
y = df['Cannabis']

label = LabelEncoder()
y = label.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



🤖 ML Models Implemented
Model	Accuracy
Logistic Regression	93%
Random Forest	      92%
Decision Tree     	91%
Naive Bayes	        88%
K-Nearest Neighbors	86%


✅ Logistic Regression
python
Copy
Edit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

lr = LogisticRegression(max_iter=200)
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred_lr))
print(classification_report(y_test, pred_lr))



🌳 Random Forest
python
Copy
Edit
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)



📏 Evaluation Metrics
python
Copy
Edit
from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, pred_rf))
print(classification_report(y_test, pred_rf))




📈 Visualization of Results
python
Copy
Edit
feat_importances = pd.Series(rf.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh', color='skyblue')



✅ Feature Importance Insights:

Impulsiveness, Sensation Seeking, and Neuroticism were among the top features.

Age and Education also played moderate roles in prediction.



📌 Project Folder Structure
css
Copy
Edit
├── README.md
├── requirements.txt
├── drug_consumption.csv
├── Drug_Consumption_Prediction.ipynb
└── src/
    ├── data_preprocessing.py
    ├── model_training.py
    └── visualizations.py


🧠 Key Learnings
1. Behavioral traits have a predictive influence on drug usage.

2. Logistic Regression proved to be the most interpretable and accurate model.

3. Cross-model comparison gives a clear performance landscape.



🔮 Future Enhancements
1. Multi-class classification for drug usage frequency

2. Deployment as a drug risk prediction web app (Streamlit/Flask)

3. Integration with mental health survey APIs

4. Adding temporal usage pattern analysis (Time Series)



▶️ Run on Google Colab
🔗 Open the Notebook in Colab (replace with actual link)


👨‍💻 Author
Vedant Singh
🎓 B.Tech IT | III Year
🏫 IIIT Sonepat
📧 svedant2103@gmail.com
🔗 LinkedIn

📜 License
This project is developed for academic and educational purposes under faculty guidance and is free for non-commercial use.

yaml
Copy
Edit

---

### ✅ Extras (Optional)

Let me know if you'd like me to:

- Generate a `requirements.txt` file from your notebook
- Add badges (build, license, etc.)
- Auto-deploy this on Hugging Face or Streamlit
- Generate visuals like confusion matrix heatmaps
