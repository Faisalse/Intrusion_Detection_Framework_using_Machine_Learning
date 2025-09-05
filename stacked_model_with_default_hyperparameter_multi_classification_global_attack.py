from preprocessing.TON_IOT_multi_classification import *
from helper_functions import *
import matplotlib.pyplot as plt
from accuracy.multi_accuracy import *

# baselines
from algorithms.DTree.DTree import * 
from algorithms.MLP.MLP import *
from algorithms.LR.lr import *
from algorithms.NB.nb import *
from algorithms.SVM.svm import *

# stacked models
from algorithms.CatB.CatB import *
from algorithms.LightB.LightB import *
from algorithms.XGBoost.XGBoost import *
from algorithms.AdaBoost.AdaBoost import *
from algorithms.GBC.GraBC import *


accuracy_objects_dict = dict()
accuracy_objects_dict["measure"] = Acc()

# import models --- > baselines
models_object_dict = dict()
models_object_dict["SVM"] = SVM()
models_object_dict["NB"] = NB()
models_object_dict["LR"] = LR()
models_object_dict["MLP"] = MLP()
models_object_dict["DT"] = DTree()

# use XGBoost as stacked model.............................................
stacked_model_dict = dict()
stacked_model_dict["XGBoost (stacked)"] = XGBoost()
stacked_model_dict["CatBoost (stacked)"] = CatB()
stacked_model_dict["LightBoost (stacked)"] = LightB()
stacked_model_dict["AdaBoost (stacked)"] = AdaBoost()
stacked_model_dict["GraBoost (stacked)"] = GBC()

DATA_PATH = r'./data/raw/'
data_name = "ToN_IoT_train_test_network"

path = Path("results/multi/defaultHyperparameters/")
path.mkdir(parents=True, exist_ok=True)

result_dict = dict()
for key in models_object_dict.keys():
    result_dict[key] = []
for key in stacked_model_dict.keys():
    result_dict[key] = []

X, y = data_load(DATA_PATH, data_name)
X_train, X_test, y_train, y_test = split_data_train_test(X, y)

attack = True
meta_features_trainX, meta_features_trainY = k_fold_return_meta_features(X_train, y_train, models_object_dict, 
                                                                            accuracy_objects_dict, path, attack=attack)
for attacked_value in [1, 0.01, 0.02, 0.03, 0.04, 0.05]:
    X_test = apply_random_scaling_df_global(X_test, ratio= attacked_value, scale = attacked_value)
    meta_features_testX, meta_features_testY, result_dataframe = return_metafeatures_for_single_splits(X_train, y_train, X_test, 
                                                                                                       y_test, models_object_dict, 
                                                                                                       accuracy_objects_dict)
    
    result = stacked_model_object_dictAND_accuracy_dict(meta_features_trainX, meta_features_trainY, meta_features_testX, 
                                           meta_features_testY, stacked_model_dict, accuracy_objects_dict )
    
    for key in models_object_dict.keys():
        result_dict[key]+=[result_dataframe[key]["measure"]]

    for key in result.keys():
        result_dict[key]+=[result[key]["measure"]]
    

markers = ['o', 's', '^', 'D', 'v', '*', 'P', 'X', '<', '>']
colors = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#7f7f7f',  # gray
    '#bcbd22',  # olive
    '#17becf'   # cyan/teal
]

x = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
i = 0
for key in result_dict.keys():
    plt.plot(x, result_dict[key], color=colors[i], marker=markers[i], linestyle='--', linewidth=2, markersize=8, 
             markerfacecolor='white', markeredgecolor='black', label= key ,alpha=0.9)
    i = i + 1    
x1 = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
plt.xticks(x, x1)
# Add labels and title
plt.ylabel('Accuracy')
plt.xlabel('Attack Strength')
plt.legend()
plt.savefig(path / "global_attack.pdf", format='pdf')
plt.show()