import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pathlib import Path
from matplotlib.lines import Line2D
from preprocessing.TON_IOT_multi_classification import *
###############################################################################
def tsne_visualization(X, y, perplexity=100, random_state=42):
    
    path = Path("results/")
    path.mkdir(parents=True, exist_ok=True)

    if Path("results/multi_tsna.csv").is_file():
        df = pd.read_csv(path / "binary_tsna.csv", sep = "\t")
    else:
        tsne = TSNE(n_components=3, perplexity=100, random_state=42)
        X_embedded = tsne.fit_transform(X)
        df = pd.DataFrame()
        df["X_component"] = X_embedded[:, 0]
        df["Y_component"] = X_embedded[:, 1]
        df["Z_component"] = X_embedded[:, 2]
        df.to_csv(path / "multi_tsna.csv", index = False, sep = "\t")
        
    y_arr = np.asarray(y).ravel()
    cats, inv = np.unique(y_arr, return_inverse=True)
    K = len(cats)

    cmap = plt.cm.get_cmap("tab10" if K <= 10 else "tab20")
    colors = [cmap(i % cmap.N) for i in range(K)]
    point_colors = [colors[i] for i in inv]

    # ---- Plot with constrained layout so legends don’t get clipped ----
    fig = plt.figure(figsize=(8, 6), constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d')
 
    sc = ax.scatter(
        df["X_component"], df["Y_component"], df["Z_component"],
        c=point_colors, alpha=0.8, edgecolors="none"
    )

    ax.set_xlabel("t-SNE Dimension X")
    ax.set_ylabel("t-SNE Dimension Y")
    ax.set_zlabel("t-SNE Dimension Z")

    # ---- Build explicit legend handles (one per class) ----
    handles = [
        Line2D([0], [0], marker='o', linestyle='None',
            markerfacecolor=colors[i], markeredgecolor='k', label=str(cats[i]))
        for i in range(K)
    ]

    # Put legend at upper-right of the *figure* to avoid 3D clipping
    fig.legend(
    handles=handles,
    title="Class labels",
    loc="upper center",
    bbox_to_anchor=(0.5, 0.3),
    ncol=len(handles),
    #fontsize=12,            # legend text
    #title_fontsize=14,       # legend title
    markerscale=1.5
)
    plt.savefig(path / "tsna_multi.pdf", format='pdf')
    plt.show()

###############################################################################

DATA_PATH = r'./data/raw/'
data_name = "ToN_IoT_train_test_network"

path = Path("results/binary/")
path.mkdir(parents=True, exist_ok=True)
X, y = data_load(DATA_PATH, data_name)
X_train, X_test, y_train, y_test = split_data_train_test(X, y)
tsne_visualization(X_train, y_train)
print("Completed")