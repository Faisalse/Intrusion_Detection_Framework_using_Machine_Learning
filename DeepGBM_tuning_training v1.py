from algorithms.DeepGBM.DeepGBM import *
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time


optimizer.maximize(
    init_points=5,   # random exploration
    n_iter=50
            # optimization iterations
)

best_params = optimizer.max
print("\n✅ Best hyperparameters found (optimized for Macro-F1):\n", best_params)

dropout = best_params["params"]["dropout"]
emb = int(best_params["params"]["emb_dim"])
epoch = int(best_params["params"]["epoch"])
hidden_dim = best_params["params"]["hidden_dim"]
lr = best_params["params"]["lr"]

# training model with optimal HPs values.....
start = time.time()

train_accu_, vali_accu_, train_loss_, vali_loss_,  acc, pre, rec, f1_ = train_deepgbm(lr, emb, hidden_dim, dropout, epoch)

end = time.time()
print(f"Time required for training: {end - start}")
print(f"Accuracy: {acc}")
print(f"Precision: {pre}")
print(f"Recall: {rec}")
print(f"F1-score: {f1_}")

# training and testing curves
train_acc = train_accu_
val_acc   = vali_accu_
epochs = range(1, len(train_acc) + 1)

# --- Style configuration (publication ready) ---
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.linewidth": 1.2,
    "grid.alpha": 0.3
})

# --- Plot ---
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_acc, 
         marker='o', linewidth=2.5, markersize=6,
         color='#1f77b4', label='Training F1-score')
plt.plot(epochs, val_acc, 
         marker='s', linewidth=2.5, markersize=6,
         color='#ff7f0e', label='Validation F1-score')

# --- Labels and title ---
plt.xlabel('Epochs')
plt.ylabel('F1-score')

# --- Grid, limits, legend ---
plt.grid(True, linestyle='--', linewidth=0.7)
plt.ylim(0.9, 1.0)
plt.xlim(1, len(epochs))
plt.legend(loc='lower right', frameon=True, fancybox=True, shadow=True, borderpad=0.8)

# --- Annotate final values ---
plt.text(epochs[-1]+0.2, train_acc[-1], f"{train_acc[-1]:.4f}", color='#1f77b4', fontsize=10)
plt.text(epochs[-1]+0.2, val_acc[-1], f"{val_acc[-1]:.4f}", color='#ff7f0e', fontsize=10)

# --- Tidy layout ---
sns.despine()
plt.tight_layout()
path = Path("results/multi/defaultHyperparameters/")
plt.savefig(path / "learning_curve.pdf", format="pdf", bbox_inches="tight", dpi=300)
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


train_loss = train_loss_
val_loss   = vali_loss_
epochs = range(1, len(train_loss) + 1)

# --- Style configuration (same as your F1 plot) ---
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.linewidth": 1.2,
    "grid.alpha": 0.3
})

# --- Plot ---
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss,
         marker='o', linewidth=2.5, markersize=6,
         color='#2ca02c', label='Training Loss')
plt.plot(epochs, val_loss,
         marker='s', linewidth=2.5, markersize=6,
         color='#d62728', label='Validation Loss')

# --- Labels and title ---
plt.xlabel('Epochs')
plt.ylabel('Loss')

# --- Grid, limits, legend ---
plt.grid(True, linestyle='--', linewidth=0.7)
plt.xlim(1, len(epochs))
plt.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, borderpad=0.8)

# --- Annotate final values ---
plt.text(epochs[-1]+0.2, train_loss[-1], f"{train_loss[-1]:.4f}", color='#2ca02c', fontsize=10)
plt.text(epochs[-1]+0.2, val_loss[-1], f"{val_loss[-1]:.4f}", color='#d62728', fontsize=10)

# --- Tidy layout and save ---
sns.despine()
plt.tight_layout()

path = Path("results/multi/defaultHyperparameters/")
path.mkdir(parents=True, exist_ok=True)   # ensure directory exists
plt.savefig(path / "loss_curve.pdf", format="pdf", bbox_inches="tight", dpi=300)
plt.show()
