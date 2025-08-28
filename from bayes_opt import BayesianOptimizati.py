from sklearn.model_selection import cross_val_score
from algorithms.DTree.DTree import *


criterion_map = {0: "gini", 1: "entropy"}
splitter_map = {0: "best", 1: "random"}
pbounds = {
    "max_depth": (1, 30),
    "criterion": (0, 1),
    "splitter": (0, 1)
}


def optimize_dtree(max_depth, criterion, splitter, X, y, cv):
    crit = criterion_map[int(round(criterion))]
    split = splitter_map[int(round(splitter))]

    model = DTree(max_depth=int(max_depth), criterion=crit, splitter=split)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    return scores.mean()

