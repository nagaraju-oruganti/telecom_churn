# Telecom Churn Predictive Model

> Objective of the project is to predict consumer of a telecom network and data service provider given the user activity characteristics. We developed various classification models and ensambled them for better predictability. The performance of the model is primarily measured on accuracy of the prediction and achieved 94.55%, however in addition, we estimated various metrics including recall, precision, and roc-auc. 

## Table of Contents

- [General Info](#general-information)
- [Technologies Used](#technologies-used)
- [Conclusions](#conclusions)

<!-- You can include any other section that is pertinent to your problem -->

## General Information

Telecom industry is an oligopoly where small number of operators serve the consumers. The churn rate is generally between 15-25% and it is largely due to relatively easy for the consumers to switch to a different service provider. For an operator, the cost to acquire a new customer is over 5-10 times more than retain an existing customer. Identifying variables driving the churn rate and then predicting the liklihood of the customer churn the operator gives an opportunity for the business to device strategic marketing and schemes/offers driving the consumer reverse the decision. 

The detaset provided in the project have over 170 features and many of them are skewed and noisy. 
- To begin with we performed EDA and feature engineering (includind data transformation).
- For baseline models, we used several classification models including logistic regression, random forest classifier, gradient boosting classifier, and xgboost classifier, to train and ensamble. The setup achieved accuracy of 94.35 percentage of churn rate prediction.
- Used optuna package and trained model hyperparameter, and repeated the above process and achieved 94.55 percentage of churn rate prediction.

Among all the features driving the churn rate are the usage of the consumer. The consumers who uses the service regularly and recharge the service monthly have more likelihood for the operator to retain the consumer. It is encouraged the operator to remain / nudge the consumer to recharge. The pschological aspect of recharge is pre-defined contract the consumer establishes with the operator and more likely for the consumer to retain with the operator.

<!-- You don't have to answer all the questions - just the ones relevant to your project. -->

## Conclusions

Build several classification models and ensambled them to predict the conusmer churn from the operator with 94.55% accuracy.

<!-- You don't have to answer all the questions - just the ones relevant to your project. -->

## Technologies Used

- numpy - version 1.23.5
- pandas - version 1.5.3
- matplotlib - version 3.7
- seaborn - version 0.12.2
- sklearn - version 1.2.1

<!-- As the libraries versions keep on changing, it is recommended to mention the version of library used in this project -->

## Contact

Created by [@@nagaraju-oruganti] - feel free to contact me!

<!-- Optional -->
<!-- ## License -->
<!-- This project is open source and available under the [... License](). -->

<!-- You don't have to include all sections - just the one's relevant to your project -->
