from algorithms.DeepGBM.DeepGBM import *
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from bayes_opt import BayesianOptimization
from catboost import CatBoostClassifier
import torch.nn as nn
import pandas as pd
import numpy as np
import torch

# original features
train_X = pd.read_csv("data/original_features_k_fold_training.csv", sep = ";")
train_Y = train_X["label"]
del train_X["label"]

test_X = pd.read_csv("data/original_features_test.csv", sep = ";")
yest_Y = test_X["label"]
del test_X["label"]

meta_features_trainX = pd.read_csv("data/training.csv", sep = ";")
meta_features_trainY = meta_features_trainX["label"]
del meta_features_trainX["label"]

meta_features_testX = pd.read_csv("data/testing.csv", sep = ";")
meta_features_testY = meta_features_testX["label"]

del meta_features_testX["label"]


meta_features_trainX = pd.concat([meta_features_trainX.reset_index(drop=True), train_X.reset_index(drop=True)], axis=1)
meta_features_testX = pd.concat([meta_features_testX.reset_index(drop=True), test_X.reset_index(drop=True)], axis=1)


X_train = np.array(meta_features_trainX)
X_val = np.array(meta_features_testX)

y_train = np.array(meta_features_trainY)
y_val = np.array(meta_features_testY)

X = X_train
y = y_train

cat_model = CatBoostClassifier(
    iterations=250,
    depth=6,
    learning_rate=0.1,
    verbose=0
)

cat_model.fit(X_train, y_train)
# Get leaf indices for each sample and tree
leaf_train = cat_model.calc_leaf_indexes(X_train)
leaf_val = cat_model.calc_leaf_indexes(X_val)

leaf_train = leaf_train.astype(np.int64)
leaf_val = leaf_val.astype(np.int64)

print("Leaf index matrix shape:", leaf_train.shape)
n_trees = leaf_train.shape[1]
n_classes = len(np.unique(y))


class TabDataset(torch.utils.data.Dataset):
    def __init__(self, X, leaf_idx, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.leaf_idx = torch.tensor(leaf_idx, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.leaf_idx[i], self.y[i]

train_ds = TabDataset(X_train, leaf_train, y_train)
val_ds = TabDataset(X_val, leaf_val, y_val)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=64)


from bayes_opt import BayesianOptimization
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score

def train_deepgbm(lr, weight_decay, emb_dim, hidden_dim, hidden_dim2, n_layers, dropout, epoch):
    # Convert types safely
    emb_dim = int(round(emb_dim))
    hidden_dim = int(round(hidden_dim))
    hidden_dim2 = int(round(hidden_dim2))
    n_layers = int(round(n_layers))
    dropout = float(dropout)
    lr = float(lr)
    weight_decay = float(weight_decay)
    epoch = int(epoch)

    class DeepGBMNet(nn.Module):
        def __init__(self, n_num, n_trees, n_classes, emb_dim, hidden_dim, hidden_dim2, n_layers, dropout):
            super().__init__()
            # Compute leaf range across train/val to avoid out-of-range errors
            leaf_all = np.vstack([leaf_train, leaf_val])
            max_leaf_ids = leaf_all.max(axis=0)

            # Embedding layers for each tree
            self.leaf_embs = nn.ModuleList([
                nn.Embedding(int(max_leaf_ids[t]) + 1, emb_dim)
                for t in range(n_trees)
            ])

            # MLP backbone
            input_dim = n_num + n_trees * emb_dim
            layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            if n_layers > 1:
                layers += [nn.Linear(hidden_dim, hidden_dim2), nn.ReLU(), nn.Dropout(dropout)]
            layers += [nn.Linear(hidden_dim2 if n_layers > 1 else hidden_dim, len(np.unique(y)))]
            self.net = nn.Sequential(*layers)

        def forward(self, x_num, leaf_idx):
            leaf_vecs = []
            for t, emb in enumerate(self.leaf_embs):
                idx = leaf_idx[:, t].clamp(0, emb.num_embeddings - 1)
                leaf_vecs.append(emb(idx))
            leaf_concat = torch.cat(leaf_vecs, dim=1)
            out = torch.cat([x_num, leaf_concat], dim=1)
            return self.net(out)

    # Initialize model and optimizer
    model = DeepGBMNet(
        n_num=X.shape[1],
        n_trees=leaf_train.shape[1],
        n_classes=len(np.unique(y)),
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        hidden_dim2=hidden_dim2,
        n_layers=n_layers,
        dropout=dropout
    )

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Quick training loop (fewer epochs for BO)
    for epoch1 in range(epoch):
        model.train()
        for xb, leafb, yb in train_loader:
            opt.zero_grad()
            out = model(xb, leafb)
            loss = criterion(out, yb)
            loss.backward()
            opt.step()

    # Evaluate on validation set → Macro-F1
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, leafb, yb in val_loader:
            logits = model(xb, leafb)
            preds = logits.argmax(1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(yb.cpu().numpy())

    val_f1 = f1_score(y_true, y_pred, average='weighted')
    return val_f1


pbounds = {
    "lr": (1e-4, 5e-3),
    "weight_decay": (1e-6, 5e-3),
    "emb_dim": (4, 24),
    "hidden_dim": (64, 256),
    "hidden_dim2": (32, 192),
    "n_layers": (1, 2),
    "dropout": (0.0, 0.5),
    "epoch": (10, 50),
}

optimizer = BayesianOptimization(
    f=train_deepgbm,
    pbounds=pbounds,
    random_state=42,
    verbose=2
)

optimizer.maximize(
    init_points=5,   # random exploration
    n_iter=50        # optimization iterations
)

best_params = optimizer.max
print("\n✅ Best hyperparameters found (optimized for Macro-F1):\n", best_params)

