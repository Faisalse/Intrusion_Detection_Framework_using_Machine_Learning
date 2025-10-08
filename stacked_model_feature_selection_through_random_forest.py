from preprocessing.TON_IOT_multi_classification import *
from helper_functions import *
from accuracy.multi_accuracy import *
import pandas as pd
import torch
from denoising_autoencoder import *
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
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
accuracy_objects_dict["Accuracy"] = Acc()
accuracy_objects_dict["Precision"] = Precision()
accuracy_objects_dict["Recall"] = Recall()
accuracy_objects_dict["F1_score"] = F1_score()

# import models --- > baselines
models_object_dict = dict()
models_object_dict["SVM"] = SVM()
models_object_dict["NB"] = NB(var_smoothing = 0.008562512157013471 )
models_object_dict["LR"] = LR(solver = "saga", penalty='none', C = 375)
models_object_dict["MLP"] = MLP(hidden_layer_sizes=(75, 57), alpha=0.08324593965363418, learning_rate_init = 0.01826431422398935, max_iter= 17)
#DT,Max_depth: 86 criterion: gini splitter: random
models_object_dict["DT"] = DTree(max_depth= 86, criterion="gini", splitter='random')

DATA_PATH = r'./data/raw/'
data_name = "ToN_IoT_train_test_network"

path = Path("results/multi/optimalhyperparameter/")
path.mkdir(parents=True, exist_ok=True)

X, y = data_load(DATA_PATH, data_name)
X_train, X_test, y_train, y_test = split_data_train_test(X, y)

rf = RandomForestClassifier(
    n_estimators=200,       # number of trees
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)


importances = pd.Series(rf.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

print(importances.head(10))  # top 10 most important features


import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
importances.head(20).plot(kind='bar')
plt.title("Top 20 Important Features (Random Forest)")
plt.ylabel("Feature Importance Score")
plt.show()

top_k = 10
selected_features = importances.head(top_k).index
X_train_sel = X_train[selected_features]
X_test_sel = X_test[selected_features]

# retrain classifier
rf_sel = RandomForestClassifier(n_estimators=200, random_state=42)
rf_sel.fit(X_train_sel, y_train)
y_pred = rf_sel.predict(X_test_sel)
print("Accuracy with selected features:", accuracy_score(y_test, y_pred))
X_train = X_train_sel
X_test = X_test_sel

meta_features_trainX, meta_features_trainY = k_fold_return_meta_features(X_train, y_train, models_object_dict, 
                                                                         accuracy_objects_dict, path)


meta_features_testX, meta_features_testY, result_dataframe = return_metafeatures_for_single_splits(X_train, y_train, X_test, 
                                                                                                       y_test, models_object_dict, 
                                                                                                       accuracy_objects_dict)

# use XGBoost as stacked model.............................................
stacked_model_dict = dict()
# CatBoost (stacked),"iterations: 135 learning_rate: 0.25270630381399795 depth: 6,  l2_leaf_reg: 1.4654355982058695"
stacked_model_dict["CatBoost"] = CatB(iterations= 135, learning_rate=0.25270630381399795, depth= 6, l2_leaf_reg = 1.4654355982058695)
stacked_model_dict["XGBoost"] = XGBoost()


stacked_model_dict["LightBoost"] = LightB()
stacked_model_dict["AdaBoost"] = AdaBoost()
stacked_model_dict["GraBoost"] = GBC()

results_stacked = stacked_model_object_dictAND_accuracy_dict(meta_features_trainX, meta_features_trainY, meta_features_testX, 
                                           meta_features_testY, stacked_model_dict, accuracy_objects_dict)

print("Print and save final results")
for key in results_stacked.keys():
    result_dataframe[str(key) + str(" (stacked)")] = results_stacked[key]
df = pd.DataFrame.from_dict(result_dataframe, orient="index")
df.to_csv(path / "results_extracted_features.txt", index = True, sep = "\t")