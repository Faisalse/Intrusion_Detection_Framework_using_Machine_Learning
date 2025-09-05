from sklearn.model_selection import cross_val_score
from algorithms.LightB.LightB import *
from sklearn.metrics import make_scorer, f1_score


lightbounds = {
    "n_estimators": (100, 500),            # Integer
    "learning_rate": (0.01, 0.3),           # Float
    "max_depth": (3, 20),                   # Integer (LightGBM can handle -1 as "no limit", but use range here)
    "num_leaves": (15, 100),                # Integer
    "min_child_samples": (5, 100)           # Integer
}

def optimize_lightb(n_estimators, learning_rate, max_depth, num_leaves, min_child_samples, X_train, y_train, X_valid, y_valid):
    model = LightB(
        n_estimators=int(n_estimators),
        learning_rate=learning_rate,
        max_depth=int(max_depth),
        num_leaves=int(num_leaves),
        min_child_samples=int(min_child_samples)
    )

    model.fit(X_train, y_train)
    accuracy = model.model.score(X_valid, y_valid)
    return accuracy
    

