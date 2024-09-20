# Hotel Booking Cancellation Prediction

## Project Overview

This project aims to predict hotel booking cancellations using machine learning techniques. By analyzing historical booking data, the model helps hotels proactively manage cancellations, optimize revenue strategies, and improve operational efficiencies.

The project utilizes data from two hotel types (a resort and a city hotel) and applies data mining and machine learning techniques to predict cancellation probabilities. By implementing a pipeline that includes feature engineering, data preprocessing, and model evaluation, the study identifies key drivers of cancellations and offers actionable insights for the hospitality industry.

## Dataset

The dataset includes bookings made between July 2015 and August 2017 for two types of hotels:

- **Resort Hotel (H1)**: 40,060 bookings with a 27.8% cancellation rate.
- **City Hotel (H2)**: 79,330 bookings with a 41.7% cancellation rate.

The data contains 31 variables including booking details, customer demographics, and hotel-specific information such as room type and special requests.

## Key Features

1. **LeadTime**: Days between booking and arrival.
2. **ADRThirdQuartileDeviation**: Deviation of Average Daily Rate (ADR) from the third quartile of comparable bookings.
3. **DepositType**: Type of deposit (non-refundable, no deposit, etc.).
4. **TotalOfSpecialRequests**: Number of special requests by the customer.

## Data Preprocessing and Feature Engineering

- **Missing data handling**: Missing values were addressed and irrelevant columns were removed.
- **Encoding**: 
  - **One-Hot Encoding** for categorical variables with limited categories.
  - **Logit-Odds Encoding** for high-cardinality variables like Agent and Company.
- **SMOTE**: Synthetic Minority Over-sampling Technique was used to handle class imbalance (cancellations vs. non-cancellations).

## Machine Learning Models

Several machine learning models were tested, including:
- Random Forest
- Logistic Regression
- AdaBoost
- Decision Tree
- K-Nearest Neighbors
- XGBoost
- Bagging
- Naive Bayes

### Model Comparison

The models were evaluated based on accuracy, precision, recall, F1-score, and ROC AUC. The Random Forest model emerged as the best performer with the following metrics:

- **Accuracy**: 0.8503
- **Precision**: 0.8150
- **Recall**: 0.7710
- **F1 Score**: 0.7924
- **ROC AUC**: 0.9119

## Fine-tuning

Using GridSearchCV, the hyperparameters for the Random Forest model were fine-tuned to maximize recall. The optimal parameters were:

- `n_estimators`: 200
- `max_depth`: None
- `max_features`: sqrt

## Graphical User Interface (GUI)

A GUI was developed to allow hotel operators to input booking details and receive real-time predictions on whether a booking is likely to be canceled. The interface is user-friendly, enabling non-technical users to utilize the model for daily operations.

## Conclusion

This project demonstrates the ability of machine learning to predict hotel booking cancellations, providing hotels with a powerful tool to optimize inventory management and revenue forecasting. The Random Forest model, fine-tuned with hyperparameter optimization, proves to be the most effective in predicting cancellations.

## References

- [Antonio, Almeida, and Nunes (2017)](https://doi.org/10.1109/ICMLA.2017.00-11): Predicting Hotel Booking Cancellations using Machine Learning.
- [Andriawan et al. (2020)](https://doi.org/10.1109/ICICoS51170.2020.9299011): Prediction of Hotel Booking Cancellation using CRISP-DM.
- [Nuno Antonio, Ana de Almeida, Luis Nunes, (2019)](https://doi.org/10.1016/j.dib.2018.11.126): Hotel booking demand datasets.
