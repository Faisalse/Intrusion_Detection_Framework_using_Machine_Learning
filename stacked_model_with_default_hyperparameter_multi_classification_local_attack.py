from preprocessing.TON_IOT_multi_classification import *
from helper_functions import *
from accuracy.multi_accuracy import *
from algorithms.RF.RF import *
from algorithms.CatB.CatB import *
from algorithms.LightB.LightB import *
from algorithms.XGBoost.XGBoost import *
from algorithms.kNN.kNN import *
import pandas as pd
from algorithms.DTree.DTree import * 
from algorithms.MLP.MLP import *
from algorithms.LR.lr import *
from algorithms.NB.nb import *

accuracy_objects_dict = dict()
accuracy_objects_dict["F1_score"] = F1_score()

# import models
models_object_dict = dict()


models_object_dict["NB"] = NB()
models_object_dict["LR"] = LR()
models_object_dict["MLP"] = MLP()
models_object_dict["DT"] = DTree()
models_object_dict["CatB"] = CatB()
models_object_dict["LightB"] = LightB()

# use XGBoost as stacked model.............................................
stacked_model_dict = dict()
Stacked_Model_name = "XGBoost"
stacked_model_dict[Stacked_Model_name] = XGBoost()

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

meta_features_trainX, meta_features_trainY = k_fold_return_meta_features(X_train, y_train, models_object_dict, 
                                                                            accuracy_objects_dict, path, attack=False)
for attacked_value in [1, 0.01, 0.02, 0.03, 0.04, 0.05]:
    X_test = apply_random_scaling_df_local(X_test, ratio= attacked_value)
    meta_features_testX, meta_features_testY, result_dataframe = return_metafeatures_for_single_splits(X_train, y_train, X_test, 
                                                                                                       y_test, models_object_dict, 
                                                                                                       accuracy_objects_dict)
    
    result = stacked_model_object_dictAND_accuracy_dict(meta_features_trainX, meta_features_trainY, meta_features_testX, 
                                           meta_features_testY, stacked_model_dict, accuracy_objects_dict )
    
    for key in models_object_dict.keys():
        result_dict[key]+=[result_dataframe[key]["F1_score"]]
    result_dict[Stacked_Model_name] += [result[Stacked_Model_name]["F1_score"]]

markers = ['o', 's', '^', 'D', 'v', '*', 'P']
colors = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
]
import matplotlib.pyplot as plt
x = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
plt.plot(x, result_dict[Stacked_Model_name], color=colors[6], marker=markers[0], linestyle='--', linewidth=2, markersize=8, markerfacecolor='white', markeredgecolor='black', label='XGBoost (stacked)',alpha=0.9)
plt.plot(x, result_dict["LightB"], color=colors[1], marker=markers[1], linestyle='--', linewidth=2, markersize=8, markerfacecolor='white', markeredgecolor='black', label='LightBoost',alpha=0.9)
plt.plot(x, result_dict["CatB"], color=colors[2], marker=markers[2], linestyle='--', linewidth=2, markersize=8, markerfacecolor='white', markeredgecolor='black', label='CatBoost',alpha=0.9)
plt.plot(x, result_dict["DT"], color=colors[3], marker=markers[3], linestyle='--', linewidth=2, markersize=8, markerfacecolor='white', markeredgecolor='black', label='DT',alpha=0.9)
plt.plot(x, result_dict["MLP"], color=colors[4], marker=markers[4], linestyle='--', linewidth=2, markersize=8, markerfacecolor='white', markeredgecolor='black', label='MLP',alpha=0.9)
plt.plot(x, result_dict["LR"], color=colors[5], marker=markers[5], linestyle='--', linewidth=2, markersize=8, markerfacecolor='white', markeredgecolor='black', label='LR',alpha=0.9)
plt.plot(x, result_dict["NB"], color=colors[0], marker=markers[6], linestyle='--', linewidth=2, markersize=8, markerfacecolor='white', markeredgecolor='black', label='NB',alpha=0.9)

x1 = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
plt.xticks(x, x1)
# Add labels and title
plt.ylabel('F1-score')
plt.xlabel('Attack Strength')
plt.legend()
plt.savefig(path / "local_attack.pdf", format='pdf')
plt.show()


