from preprocessing.TON_IOT_multi_classification import *
from helper_functions import *
from accuracy.multi_accuracy import *
import pandas as pd
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from helfer_functions.import_models import *


DATA_PATH = r'./data/raw/'
data_name = "ToN_IoT_train_test_network"

path = Path("results/multi/defaultHyperparameters/")
path.mkdir(parents=True, exist_ok=True)

X, y = data_load(DATA_PATH, data_name)
X_train, X_test, y_train, y_test = split_data_train_test(X, y)








accuracy_objects_dict = dict()
accuracy_objects_dict["Accuracy"] = Acc()
accuracy_objects_dict["Precision"] = Precision()
accuracy_objects_dict["Recall"] = Recall()
accuracy_objects_dict["F1_score"] = F1_score()

# import models --- > baselines
models_object_dict = dict()
models_object_dict["DT"] = DTree(max_depth= 86, criterion="gini", splitter='random')
# CatBoost (stacked),"iterations: 135 learning_rate: 0.25270630381399795 depth: 6,  l2_leaf_reg: 1.4654355982058695"
models_object_dict["CatBoost"] = CatB(iterations= 150, learning_rate = 0.25270630381399795, depth = 6, l2_leaf_reg = 1.4654355982058695)
# XGBoost (stacked),n_estimators: 380 max_depth: 15 subsample: 0.9829727315201009 
# colsample_bytree: 0.9821163261855509  gamma: 2.754057872914979 reg_alpha: 9.293502339665233  reg_lambda: 9.04947413810599
models_object_dict["XGBoost"] = XGBoost(n_estimators=380, max_depth = 15, subsample = 0.9829727315201009,
                                        colsample_bytree = 0.9821163261855509, gamma = 15, reg_alpha = 9.293502339665233,  reg_lambda = 9.04947413810599)
# LightBoost (stacked),"n_estimators: 185 learning_rate: 0.01596950334578271 max_depth: 19,  
# num_leaves: 30 min_child_samples: 84"
models_object_dict["LightBoost"] = LightB(n_estimators = 185, learning_rate = 0.01596950334578271, max_depth = 19, num_leaves = 30, min_child_samples = 84)

# AdaBoost (stacked),n_estimators: 200 learning_rate: 0.07286124900578073
models_object_dict["AdaBoost"] = AdaBoost(n_estimators= 200, learning_rate = 0.07286124900578073)
# GraBoost (stacked),"n_estimators: 146 learning_rate: 0.01596950334578271 
# max_depth: 15,  min_samples_leaf: 42 subsample: 0.5909124836035503"
models_object_dict["GraBoost"] = GBC(n_estimators= 146, learning_rate = 0.01596950334578271,
                                     max_depth = 15, min_samples_leaf = 42, subsample = 0.5909124836035503)



meta_features_trainX, meta_features_trainY, original_features_k_fold = k_fold_return_meta_features(X_train, y_train, models_object_dict, 
                                                                         accuracy_objects_dict, path)

# Assign column names.............
label = original_features_k_fold["label"]
del original_features_k_fold["label"]
original_features_k_fold.columns = X.columns
original_features_k_fold["label"] = label


meta_features_testX, meta_features_testY, result_dataframe = return_metafeatures_for_single_splits(X_train, y_train, X_test, 
                                                                                                       y_test, models_object_dict, 


                                                                                                       accuracy_objects_dict)
result_dataframe = pd.DataFrame(result_dataframe)
result_dataframe = result_dataframe.transpose()

result_dataframe = result_dataframe.sort_values(by ="F1_score")
result_dataframe.to_csv(path / "baseline_results.csv", sep = "\t")
# save meta_features..................
meta_features_trainX["label"] = meta_features_trainY
meta_features_testX["label"] = meta_features_testY

meta_features_trainX.to_csv(path / "training_meta.csv", sep = ";", index = False)
meta_features_testX.to_csv(path / "testing_meta.csv", sep = ";", index = False)
original_features_k_fold.to_csv(path / "original_features_k_fold_training.csv", sep = ";", index = False)
X_test["label"] = y_test
X_test.to_csv(path / "original_features_test.csv", sep = ";", index = False)