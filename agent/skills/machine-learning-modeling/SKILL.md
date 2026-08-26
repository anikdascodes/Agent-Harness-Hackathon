---
name: machine-learning-modeling
description: Instructions for benchmarking ML models in sandbox, evaluating generalization metrics, extracting feature importances, and ranking models.
---

# Machine Learning Modeling & Evaluation Protocol

## Objectives:
1. Run automated, leak-free machine learning benchmarks using `benchmark_ml_models`.
2. Compare 4 diverse model architectures (Random Forest, Gradient Boosting, Logistic Regression, Decision Tree).
3. Evaluate on unseen test data with appropriate metric weighting (ROC-AUC / F1 for imbalanced classification, RMSE / R2 for regression).
4. Extract the top 8 predictive drivers with business explanations.

## Translation Guide for Business Owners:

| Technical ML Term | Plain-English Business Translation |
| :--- | :--- |
| **ROC-AUC of 0.85** | *"The model correctly ranks an at-risk customer above a safe customer 85% of the time."* |
| **Feature Importance of 0.35 on `support_tickets`** | *"Support ticket volume is the single strongest predictor of whether a customer will cancel."* |
| **Precision of 80% at 70% Recall** | *"Out of every 10 customers the model flags as churning, 8 actually churn, catching 70% of all potential cancellations."* |
| **MAE of $120 on Revenue Forecast** | *"On average, our sales predictions are within $120 of actual daily store revenue."* |
