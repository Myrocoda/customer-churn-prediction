# Customer Churn Prediction: Random Forest vs SVM

Predicts whether a bank customer will leave (`Exited`) using the Churn_Modelling dataset (10,000 customers, 14 columns), and compares two classifiers.

## Pipeline
- Dropped non-predictive columns (`RowNumber`, `Surname`, `CustomerId`)
- Label-encoded `Gender`; one-hot encoded `Geography` (`drop_first=True`)
- 80/20 stratified train/test split (`random_state=42`)
- Standardised features for the SVM (Random Forest uses raw features)

## Models
- `RandomForestClassifier(n_estimators=200)`
- `SVC(kernel='rbf')`

## Results (test set, 2,000 customers)
| Model | Accuracy | Churn precision | Churn recall | Churn F1 |
|---|---|---|---|---|
| Random Forest | 0.865 | 0.79 | 0.46 | 0.58 |
| SVM (RBF) | 0.861 | 0.85 | 0.38 | 0.53 |

Both models score similarly on accuracy, but recall on the churn class is low (the dataset is ~20% churners and no imbalance handling was applied). Random Forest catches more churners; SVM is more precise when it flags one.

## Run it
```bash
pip install -r requirements.txt
jupyter notebook Customer_Churn.ipynb
```

## Files
- `Churn_Modelling.csv`: dataset
- `Customer_Churn.ipynb`: analysis and modelling

## Next steps
Class weighting or SMOTE, threshold tuning, hyperparameter search, and feature-importance analysis.
