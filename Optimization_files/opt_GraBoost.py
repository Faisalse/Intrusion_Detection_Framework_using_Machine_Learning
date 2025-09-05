from algorithms.GBC.GraBC import * 


gbcbounds = {
    "n_estimators": (50, 500),           # Integer
    "max_depth": (3, 15),                  # Integer
    "learning_rate": (0.01, 0.3),          # Float
    "subsample": (0.5, 1.0),
    "min_samples_leaf": (5, 50)
                                  # Float
}


def optimize_model(n_estimators, max_depth, learning_rate, subsample, min_samples_leaf, X_train, y_train, X_valid, y_valid):
    
    model = GBC(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        learning_rate = float(learning_rate),
        subsample = float(subsample),
        min_samples_leaf = int(min_samples_leaf)
        )
    
    model.fit(X_train, y_train)
    accuracy = model.model.score(X_valid, y_valid)
    return accuracy




