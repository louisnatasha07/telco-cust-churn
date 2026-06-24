# Telco Customer Churn Prediction

This project is a final project for PPDKM/Kapita Selekta. The project focuses on predicting customer churn in a telecommunication company using data mining classification algorithms and the CRISP-DM approach.

## Project Overview

Customer churn is a condition where customers stop using a service or move to another provider. In the telecommunication industry, churn prediction is important because it helps companies identify customers who are likely to leave. By knowing potential churn customers, companies can prepare retention strategies such as special offers, service improvement, or personalized communication.

This project uses the Telco Customer Churn dataset and applies several machine learning algorithms to compare model performance.

## Methodology

The project follows the CRISP-DM framework:

1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation
6. Deployment

## Algorithms Used

The classification algorithms used in this project are:

* Logistic Regression
* K-Nearest Neighbor
* Decision Tree
* Random Forest
* AdaBoost
* Gradient Boosting

## Best Model

Based on the evaluation result, the best model is Decision Tree because it achieved the highest F1-score and recall compared to the other models.

## Evaluation Metrics

The model performance was evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix

## Deployment

The best model was deployed using Streamlit. The application allows users to input customer data and predict whether the customer is likely to churn or not.

## Files

* `streamlit_app.py` : Streamlit application file
* `telco_churn_best_model.pkl` : Saved best machine learning model
* `feature_schema.json` : Feature schema for model input
* `requirements.txt` : Required Python libraries
* `PPDKM_Telco_Customer_Churn_CRISP_DM.ipynb` : Jupyter Notebook for analysis and modeling
* `Telco-Customer-Churn.csv` : Dataset

## How to Run Locally

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run streamlit_app.py
```

or:

```bash
py -m streamlit run streamlit_app.py
```

## Author

Louis Natasha Voudy Nanlessy
L0223026
Program Studi Sains Data
Universitas Sebelas Maret
