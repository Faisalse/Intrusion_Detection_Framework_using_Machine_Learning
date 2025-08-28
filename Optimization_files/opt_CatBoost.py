from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score
from algorithms.CatB.CatB import *


catbounds = {
    "iterations": (100, 500),          # Integer, e.g. 100, 1000
    "learning_rate": (0.01, 0.3),       # Float
    "depth": (3, 10),                   # Integer
    "l2_leaf_reg": (1, 10)              # Float or Integer
}



def optimize_catb(iterations, learning_rate, depth, l2_leaf_reg, X, y, cv):
    model = CatB(
        iterations=int(iterations),
        learning_rate=learning_rate,
        depth=int(depth),
        l2_leaf_reg=l2_leaf_reg
    )
    # Cross-validation with F1-macro scoring
    f1_macro = make_scorer(f1_score, average='macro')
    scores = cross_val_score(model, X, y, cv=cv, scoring=f1_macro)
    return scores.mean()

