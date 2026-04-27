# Employee Attrition Analysis & Prediction

A machine learning project to analyze employee attrition patterns and build predictive models to identify employees likely to leave an organization.

## Project Overview

Employee attrition is a major business problem affecting workforce stability and productivity. This project performs end-to-end data analysis and machine learning to understand and predict employee attrition.

## Objectives

- Analyze employee attrition patterns using HR data
- Perform exploratory data analysis (EDA)
- Identify key factors influencing attrition
- Build machine learning classification models
- Compare multiple models (Random Forest and XGBoost)
- Evaluate performance using standard metrics
- Extract actionable business insights

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Jupyter Notebook

## Project Structure

Employee-Attrition-Analysis/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_model.ipynb
│   ├── 03_model_comparison.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── eda.py
│   ├── utils.py
│   ├── models/
│   │   ├── random_forest.py
│   │   ├── xgboost_model.py
│   │   ├── tuning.py
│   │   └── evaluate.py
│
├── outputs/
│   ├── plots/
│   ├── reports/
│
├── main.py
├── requirements.txt
└── README.md

## Workflow

1. Data Collection  
2. Data Cleaning and Preprocessing  
3. Exploratory Data Analysis (EDA)  
4. Feature Engineering  
5. Model Training  
   - Random Forest  
   - XGBoost  
6. Model Evaluation and Comparison  
7. Insights Generation  

## Model Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1 Score  
- ROC-AUC Score  
- Precision-Recall Curve  

## Key Insights

- Overtime is one of the strongest predictors of attrition  
- Younger employees show higher attrition probability  
- Job satisfaction significantly impacts retention  
- Certain departments show higher turnover rates  

## Future Improvements

- Deploy using Streamlit dashboard  
- Add SHAP-based model explainability  
- Hyperparameter tuning using Optuna  
- Build real-time prediction API  

## How to Run This Project

git clone https://github.com/your-username/Employee-Attrition-Analysis.git  
cd Employee-Attrition-Analysis  
pip install -r requirements.txt  
python main.py  

## Author

Hiya Dutta  
B.Tech Computer Science and Engineering  
KIIT University  

## Note

This project demonstrates a complete machine learning workflow including data analysis, model building, evaluation, and business insight generation.sitory a star ⭐ on GitHub!
