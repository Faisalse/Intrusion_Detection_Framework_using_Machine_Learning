from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score

# Mapping for kernels (only linear and rbf)
kernel_map = {0: "linear", 1: "rbf"}

# Mapping for discrete C values
C_map = {0: 0.001, 1: 0.01, 2: 0.1, 3: 1, 4: 10, 5: 100}

# Bounds for optimizer
svm_bounds = {
    "C": (0, 5),        # index in C_map
    "kernel": (0, 1)    # index in kernel_map
}



def optimize_svm(C, kernel, X, y, cv):
    # Map discrete values
    C_value = C_map[int(round(C))]
    ker = kernel_map[int(round(kernel))]

    # Build SVM model
    model = SVC(C=C_value, kernel=ker, probability=True)

    # Cross-validation with F1-macro scoring
    f1_macro = make_scorer(f1_score, average='macro')
    scores = cross_val_score(model, X, y, cv=cv, scoring=f1_macro)
    return scores.mean()
