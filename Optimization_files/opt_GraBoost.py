from sklearn.model_selection import cross_val_score
from algorithms.XGBoost.XGBoost import *


xgbounds = {
    "n_estimators": (50, 500),           # Integer
    "max_depth": (3, 15),                  # Integer
    "learning_rate": (0.01, 0.3),          # Float
    "subsample": (0.5, 1.0),               # Float
    "colsample_bytree": (0.5, 1.0),        # Float
    "gamma": (0, 10),                      # Float
    "reg_alpha": (0, 10),                  # Float
    "reg_lambda": (0, 10)                  # Float
}


def optimize_model(n_estimators, max_depth, learning_rate, subsample, colsample_bytree,
                   gamma, reg_alpha, reg_lambda,
                   X_train, y_train, X_valid, y_valid):
    
    model = XGBoost(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda
    )
    
    model.fit(X_train, y_train)
    accuracy = model.model.score(X_valid, y_valid)
    return accuracy




