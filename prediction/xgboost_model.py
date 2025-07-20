import pandas as pd
import sqlite3
import numpy as np
import optuna
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import f1_score, mean_squared_error, accuracy_score, classification_report
import shap
import matplotlib.pyplot as plt

#get data
conn = sqlite3.connect('data/new_data.db')   
data = pd.read_sql_query("SELECT * FROM eps_data", conn)
conn.close()

#X is all features except target column
y = data['target']
X = data.drop('target', axis=1)


#missing values = 0 and one-hot encode categories 
X = pd.get_dummies(X)
X = X.fillna(0).astype('float32')

#alignment --> remove if data indexes match
X, y = X.align(y, join='inner', axis=0)


is_classification = y.nunique() <= 10 and y.dtype in [int, bool]

#80/20 split and stratify for classification 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
    stratify=y if is_classification else None
)

#optuna hyperparameter tuning
def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 5, 20),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 15),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 15),
        'random_state': 42,
        'verbosity': 0
    }

    if is_classification:
        
        params['objective'] = 'binary:logistic' if y.nunique() == 2 else 'multi:softmax'
        if y.nunique() > 2:
            params['num_class'] = y.nunique()
        model = xgb.XGBClassifier(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) #k = 5 folds
    else:
        
        params['objective'] = 'reg:squarederror'
        model = xgb.XGBRegressor(**params)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

    scores = []
    # cross val
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_train_cv, X_val_cv = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model.fit(X_train_cv, y_train_cv, eval_set=[(X_val_cv, y_val_cv)], verbose=False)
        y_pred = model.predict(X_val_cv)

        if is_classification:
            # f1
            score = f1_score(y_val_cv, y_pred, average='macro')
        else:
            # mse
            score = mean_squared_error(y_val_cv, y_pred)

        scores.append(score)



    # maximize if classficaiton, minimize if regression
    return np.mean(scores) if not is_classification else -np.mean(scores)


study = optuna.create_study(direction='minimize' if not is_classification else 'maximize')
study.optimize(objective, n_trials=50)


print("top hyperparameters:")
print(study.best_params)

# final model with best hyperparameters 
best_params = study.best_params
best_params.update({'random_state': 42, 'verbosity': 0})

if is_classification:
    best_params['objective'] = 'binary:logistic' if y.nunique() == 2 else 'multi:softmax'
    if y.nunique() > 2:
        best_params['num_class'] = y.nunique()
    model = xgb.XGBClassifier(**best_params)
else:
    best_params['objective'] = 'reg:squarederror'
    model = xgb.XGBRegressor(**best_params)

model.fit(X_train, y_train)



# test set
y_pred = model.predict(X_test)



# scores
if is_classification:
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"Test accuracy: {acc:.4f}")
    print(f"Test F1: {f1:.4f}")
    print(classification_report(y_test, y_pred))
else:
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Test RMSE: {rmse:.4f}")


model.save_model("prediction/xgb_model.json")


#shap and features

plt.figure(figsize=(10, 6))
xgb.plot_importance(model, max_num_features=15)
plt.title("TOP FEATURES")
plt.show()


explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)
shap.plots.bar(shap_values)
