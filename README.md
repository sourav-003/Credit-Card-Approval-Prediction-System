# Credit Card Approval Prediction System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost-orange)
![Imbalanced Data](https://img.shields.io/badge/Imbalanced-SMOTE-green)
![Deployment](https://img.shields.io/badge/Deployment-Gradio-red)

## Project Overview

Banks and financial institutions receive thousands of credit card applications daily. Manually analyzing these is error-prone and time-consuming. This project automates the credit card approval process using Machine Learning to predict whether an applicant is a **"Good"** or **"Bad"** credit risk.

The solution handles severe class imbalance using **SMOTE** and utilizes an **XGBoost Classifier** for high-performance predictions. A user-friendly web interface is deployed using **Gradio**.

## Dataset

The project typically uses data containing applicant details (gender, income, education, etc.) and their credit history.
* **Input:** Applicant demographics and financial history.
* **Target:** Risk status (0 = Approved/Good, 1 = Rejected/Risk).

## Tech Stack

* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn, Plotly
* **Machine Learning:** Scikit-Learn, XGBoost, Imbalanced-learn (SMOTE)
* **Model Deployment:** Gradio, Joblib

## Methodology

### 1. Exploratory Data Analysis (EDA)
* Analyzed distributions of Income, Age, and Employment.
* Visualized correlation matrices to remove highly correlated features.
* Handled missing values and encoded categorical variables.

### 2. Handling Class Imbalance (SMOTE)
* The dataset was heavily imbalanced (more approved clients than rejected ones).
* **Technique:** Synthetic Minority Over-sampling Technique (SMOTE).
* **Critical Step:** SMOTE was applied **only to the training set** after splitting to prevent data leakage.

### 3. Model Selection & Training
Three models were evaluated:
* **Logistic Regression** (Baseline)
* **Random Forest Classifier**
* **XGBoost Classifier** (Best Performer)

### 4. Evaluation Metrics
* **ROC-AUC Score:** Used as the primary metric due to class imbalance.
* **Confusion Matrix:** To visualize False Positives vs. False Negatives.
* **Feature Importance:** Identified that `Age`, `Years of Experience`, and `Income` are the top drivers for credit decisions.

## App Demo

**Inputs:**
* Applicant Gender, Car/Property Ownership
* Income, Age, Family Size
* Job Type, Education Level, Housing Type

**Output:**
* **Decision:** Approved / Rejected
* **Risk Probability:** Percentage risk of default.

![App Interface Screenshot](screenshots/app_demo.png)

##  Results

* **Best Model:** XGBoost
* **Training Accuracy:** ~95%
* **Test ROC-AUC Score:** >0.85

The model successfully identifies high-risk candidates while maintaining a low false rejection rate.

