
# Credit Card Fraud Detection

## Overview

For this project, I wanted to see how different machine learning models perform when trying to identify fraudulent credit card transactions.

I worked with a dataset containing transaction information such as transaction amount, number of declines, whether the transaction was foreign, whether it came from a high-risk country, and chargeback information. I cleaned the data, explored the relationships between the variables, and then trained several classification models to compare their performance.

The main goal was not just to get the highest accuracy, but to see how different models handle fraud detection and the tradeoff between precision and recall.

## Dataset

The dataset contains 3,075 transactions and 12 columns.

Some of the main features include:

* Average amount per transaction per day
* Transaction amount
* Number of declined transactions per day
* Whether the transaction was foreign
* Whether the transaction came from a high-risk country
* Daily average chargeback amount
* Six-month average chargeback amount
* Six-month chargeback frequency
* Whether the transaction was fraudulent

The `Transaction date` column was removed because it did not contain usable values, and `Merchant_id` was also removed before modeling.

## Data Preparation

I first looked through the dataset to check its structure, statistics, and missing values. There were no missing values in the variables used for the analysis.

The categorical variables were originally represented using `Y` and `N`. I converted these into binary values:

* `Y` → `1`
* `N` → `0`

This included the target variable, `isFradulent`, as well as the transaction-related categorical features.

I also created a correlation matrix to get a better idea of how the different variables were related to each other and to fraud. For example, `isHighRiskCountry` and `Total Number of declines/day` had relatively strong correlations with the fraud label in this dataset.

## Models

I tried several different approaches to compare how they performed:

### Logistic Regression

I started with logistic regression as a simple baseline classification model. I standardized the features and used an 80/20 train-test split.

### Random Forest

I then used a Random Forest classifier to see how a tree-based model compared with logistic regression. I also looked at feature importance to get an idea of which variables the model relied on most.

### Balanced Bagging + Random Forest

Since fraud detection can involve an imbalance between fraudulent and non-fraudulent transactions, I also experimented with balanced bagging using a Random Forest estimator.

### SMOTE + Random Forest

I used SMOTE to create additional examples of the minority class in the training data. This gave the model a more balanced set of examples to learn from.

Before SMOTE, the training data had 2,115 non-fraudulent transactions and 345 fraudulent transactions. After applying SMOTE, both classes had 2,115 examples.

### XGBoost

I also tested XGBoost with a small set of manually selected parameters, including a maximum depth of 4, a learning rate of 0.05, and 250 estimators.

### Support Vector Machine

Finally, I trained an SVM using an RBF kernel and standardized features. I also experimented with different kernels and C values using GridSearchCV.

## Results

Here are the results I recorded from the different models:

| Model                            | Accuracy | Precision | Recall | F1 Score |
| -------------------------------- | -------: | --------: | -----: | -------: |
| Logistic Regression              |     0.99 |      0.98 |   0.96 |     0.97 |
| Random Forest                    |     0.99 |      0.99 |   0.94 |     0.97 |
| Balanced Bagging + Random Forest |     0.97 |      0.86 |   0.99 |     0.92 |
| SMOTE + Random Forest            |     0.96 |      0.83 |   0.99 |     0.90 |
| XGBoost                          |     0.99 |      0.99 |   0.92 |     0.95 |
| SVM                              |     0.99 |      0.98 |   0.96 |     0.97 |

For example, the XGBoost model had an accuracy of 0.99, precision of 0.99, recall of 0.92, and an F1 score of 0.95 on its test set.

The balanced models were interesting because they were able to identify almost all of the fraudulent transactions, with recall around 0.99. However, their precision was lower. In other words, they caught more fraud but also classified more legitimate transactions as fraudulent.

That was one of the biggest things I took away from the project: accuracy alone isn't enough for a fraud detection problem. Depending on the situation, missing a fraudulent transaction and incorrectly flagging a legitimate one can have very different consequences.

## Evaluation

I mainly used:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* ROC/AUC
* Precision-Recall curves

I focused especially on precision and recall because of the importance of correctly identifying fraudulent transactions.

## Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Imbalanced-learn

## What I Learned

This project gave me more experience with the full machine learning workflow rather than just training one model.

Some of the main things I practiced were:

* Cleaning and preparing a dataset
* Converting categorical variables into numerical values
* Exploring correlations between features
* Splitting data into training and testing sets
* Feature scaling
* Training multiple classification models
* Handling class imbalance with SMOTE and balanced bagging
* Comparing models using more than just accuracy
* Interpreting confusion matrices and classification metrics

Most importantly, I got a better understanding of how the "best" model depends on what you're actually trying to accomplish. A model with slightly lower accuracy can still be useful if it does a better job of catching the cases that matter most.

## Future Improvements

If I continued working on this project, I would:

* Clean up and standardize the train/test process across all of the models
* Tune the models more systematically
* Compare additional classification algorithms
* Explore additional feature engineering
* Use cross-validation more consistently
* Improve the handling of class imbalance
* Look more closely at which features contribute most to fraud predictions
* Build a simple interface that could be used to make predictions on new transactions
