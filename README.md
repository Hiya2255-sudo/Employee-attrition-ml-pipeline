# 👨‍💼 Employee Attrition ML Pipeline

## 📌 Project Overview
This project is an end-to-end machine learning pipeline designed to analyze and predict employee attrition using HR data. The goal is to identify key factors influencing employee turnover and provide data-driven insights to improve employee retention strategies.

---

## 🎯 Problem Statement
Employee attrition is a major challenge for organizations, leading to increased hiring costs and loss of skilled talent. This project aims to analyze HR data to understand attrition patterns, identify key drivers, and build a machine learning model to predict whether an employee will leave the company.

---

## 📊 Dataset
- IBM HR Analytics Employee Attrition Dataset (Kaggle)
- Features include:
  - Age, Gender, Department
  - Job Role, Monthly Income
  - Job Satisfaction, Overtime
  - Work Experience, Education

---

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Joblib

---

## 🔄 Project Workflow
1. Data Collection  
2. Data Cleaning & Preprocessing  
3. Exploratory Data Analysis (EDA)  
4. Feature Engineering  
5. Model Building (Classification)  
6. Model Evaluation  
7. Insights & Interpretation  

---

## 📁 Project Structure
Employee-Attrition-Analysis/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_model_building.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── eda.py
│   ├── model.py
│   ├── utils.py
│
├── models/
│   └── attrition_model.pkl
│
├── main.py
├── requirements.txt
└── README.md

---

## 🤖 Machine Learning Model
- Algorithm: Random Forest Classifier  
- Task: Binary Classification (Attrition: Yes/No)

### Evaluation Metrics:
- Accuracy  
- Precision  
- Recall  
- F1-score  

---

## 📊 Key Insights
- Employees working overtime are more likely to leave  
- Lower monthly income increases attrition risk  
- Job satisfaction strongly impacts retention  
- Certain departments show higher attrition rates  

---

## 🚀 How to Run This Project

### 1. Clone the repository
git clone https://github.com/Hiya2255-sudo/Employee-attrition-ml-pipeline.git  
cd Employee-attrition-ml-pipeline  

### 2. Install dependencies
pip install -r requirements.txt  

### 3. Run the pipeline
python main.py  

---

## 📈 Results
- Built an end-to-end ML pipeline for employee attrition prediction  
- Identified key HR factors affecting employee turnover  
- Created reusable modular code structure  

---

## 💡 Future Improvements
- Deploy using Streamlit or Flask  
- Add SHAP explainability for model interpretation  
- Improve model with XGBoost / LightGBM  
- Build real-time prediction dashboard  

---

## 👨‍💻 Author
Hiya Dutta  

---

## ⭐ If you like this project
Give this repository a star ⭐ on GitHub!
