from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score


# Mapping for discrete C values
C_map = {0: 0.001, 1: 0.01, 2: 0.1, 3: 1, 4: 10}

# Bounds for optimizer
svm_bounds = {
    "C": (0, 4)
}



def optimize_svm(C, X, y, cv):
    # Map discrete values
    C_value = C_map[int(round(C))]

    # Build SVM model
    model = SVC(C=C_value, probability=True)

    # Cross-validation with F1-macro scoring
    f1_macro = make_scorer(f1_score, average='macro')
    scores = cross_val_score(model, X, y, cv=cv, scoring=f1_macro)
    return scores.mean()
