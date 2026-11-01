
# Customer Churn Prediction Using Machine Learning

## Project Overview
This project predicts whether a telecom customer is likely to leave a company's services using Machine Learning techniques.

## Objectives
- Analyze customer churn patterns.
- Clean and preprocess customer data.
- Perform exploratory data analysis.
- Train and evaluate Machine Learning models.
- Identify important features influencing churn.
- Predict customer churn using trained models.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

## Machine Learning Algorithms
1. Logistic Regression
2. Random Forest Classifier

## Project Workflow
1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis
4. Feature Engineering
5. Model Training
6. Model Evaluation
7. Feature Importance Analysis
8. Model Comparison
9. Customer Churn Prediction

## Model Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

## Model Results

| Model | Accuracy |
|---|---:|
| Logistic Regression | 80.70% |
| Random Forest | 77.43% |

See `reports/model_comparison.csv` for the complete evaluation results.

## Feature Importance
The project generates a feature importance report at:

`reports/feature_importance.csv`

## Project Structure

```text
Customer-Churn-Prediction/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── reports/
├── src/
├── README.md
└── requirements.txt
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run model comparison:

```bash
python src/model_comparison.py
```

Run customer prediction:

```bash
python src/predict_customer.py
```

## Conclusion
This project demonstrates how Machine Learning can help telecom businesses identify customers at risk of leaving their services and support customer retention strategies.

## Author
BSc Artificial Intelligence and Machine Learning Student
