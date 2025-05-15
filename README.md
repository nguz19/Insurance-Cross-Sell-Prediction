# Insurance Cross-Sell Prediction

## Overview
This project aims to predict the likelihood of healthcare insurance customers to purchase vehicle insurance using machine learning models. By analyzing historical customer data, including demographic and behavioral features, we developed predictive models to identify potential cross-sell targets.

## Methodology / Tools and Techniques
- Data Source: Kaggle
- Preprocessing: MinMaxScaler, SMOTE for class imbalance, dummy encoding
- Models: Random Forest, Gradient Boosting
- Evaluation Metrics: Accuracy, Precision, Recall, AUC

## Key Findings
- Random Forest achieved 90% accuracy and AUC of 0.97, outperforming Gradient Boosting.
- Key predictors: Age, Previous Insurance, Annual Premium, Vehicle Damage History, Vintage.
- Gradient Boosting had slightly lower accuracy but higher recall for high-risk customers.

## Discussion
Random Forest was particularly effective in identifying cross-sell opportunities due to its robustness against noise and ability to handle imbalanced datasets. Future work includes advanced hyperparameter tuning and incorporating additional customer attributes.
