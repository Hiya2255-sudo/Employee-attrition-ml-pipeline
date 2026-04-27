
#eda.py            Data visualization and insights

import matplotlib.pyplot as plt
import seaborn as sns

def plot_attrition(df):
    sns.countplot(x='Attrition', data=df)
    plt.title("Attrition Distribution")
    plt.show()

def dept_attrition(df):
    if 'Department' in df.columns:
        sns.countplot(x='Department', hue='Attrition', data=df)
        plt.xticks(rotation=45)
        plt.show()

def salary_vs_attrition(df):
    if 'MonthlyIncome' in df.columns:
        sns.boxplot(x='Attrition', y='MonthlyIncome', data=df)
        plt.show()