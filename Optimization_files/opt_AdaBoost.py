from sklearn.model_selection import cross_val_score
from algorithms.AdaBoost.AdaBoost import *


adaboost_search_space = {
    "n_estimators": (50, 800),        # int
    "learning_rate": (1e-3, 2.0)
}


def optimize_adaboost(n_estimators, learning_rate,
                      X_train, y_train, X_valid, y_valid):
    model = AdaBoost(
        n_estimators=int(n_estimators),
        learning_rate=float(learning_rate)
    )
    model.fit(X_train, y_train)
    return model.model.score(X_valid, y_valid)  # accuracy




