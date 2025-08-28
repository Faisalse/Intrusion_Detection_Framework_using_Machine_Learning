from sklearn.model_selection import cross_val_score
from algorithms.MLP.MLP import *



lr_map = {0: 'constant', 1: 'invscaling', 2: 'adaptive'}


MLPpbounds = {
    'units1': (20, 200),                    # hidden_layer_sizes[0]
    'units2': (10, 100),                    # hidden_layer_sizes[1]               
    'alpha': (1e-5, 1e-1),                  # L2 penalty
    'learning_rate': (0, 2),                # 0: 'constant', 1: 'invscaling', 2: 'adaptive'
    'learning_rate_init': (1e-4, 1e-1),     # initial learning rate
    'max_iter': (10, 50)                   # epochs
}


from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import cross_val_score

def optimize_mlp(units1, units2, alpha, learning_rate, learning_rate_init, max_iter,
                 X, y, cv):
    
    hidden_layers = (int(units1), int(units2))
    lr_str = lr_map[int(round(learning_rate))]

    model = MLP(
        hidden_layer_sizes=hidden_layers,
        alpha=alpha,
        learning_rate=lr_str,
        learning_rate_init=learning_rate_init,
        max_iter=int(max_iter)
    )

    # Cross-validation with F1-macro scoring
    f1_macro = make_scorer(f1_score, average='macro')
    scores = cross_val_score(model, X, y, cv=cv, scoring=f1_macro)

    return scores.mean()





