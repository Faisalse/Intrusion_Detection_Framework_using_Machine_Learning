from preprocessing.TON_IOT_multi_classification import *
from helper_functions import *
from accuracy.multi_accuracy import *
import pandas as pd
from denoising_autoencoder import *
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

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
models_object_dict["MLP"] = MLP(hidden_layer_sizes=(10, 5), alpha=0.08324593965363418, learning_rate_init = 0.01826431422398935, max_iter= 17)
#DT,Max_depth: 86 criterion: gini splitter: random
models_object_dict["DT"] = DTree(max_depth= 86, criterion="gini", splitter='random')

DATA_PATH = r'./data/raw/'
data_name = "ToN_IoT_train_test_network"

path = Path("results/multi/optimalhyperparameter/")
path.mkdir(parents=True, exist_ok=True)

X, y = data_load(DATA_PATH, data_name)
X_train, X_test, y_train, y_test = split_data_train_test(X, y)



clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Make prediction on the testing data
y_pred = clf.predict(X_test)

# Classification Report
print("Classification report")
print(classification_report(y_pred, y_test))

explainer = shap.Explainer(clf)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)

shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
fig = plt.gcf()
fig.set_size_inches(8, 6)

plt.savefig(path /"shaply_summary.pdf", format="pdf", bbox_inches="tight")
plt.close(fig)

shap.summary_plot(shap_values[0], X_test)
shap.decision_plot(explainer.expected_value[0], shap_values[0], X_test.columns, ignore_warnings=True)

nu_features = 4
top_features = (
    pd.Series(clf.feature_importances_, index=X_train.columns)
      .sort_values(ascending=False)
      .head(nu_features)
      .index
      .tolist()
)
X_train = X_train[top_features]
X_test = X_test[top_features]



meta_features_trainX, meta_features_trainY = k_fold_return_meta_features(X_train, y_train, models_object_dict, 
                                                                         accuracy_objects_dict, path)


meta_features_testX, meta_features_testY, result_dataframe = return_metafeatures_for_single_splits(X_train, y_train, X_test, 
                                                                                                       y_test, models_object_dict, 
                                                                                                       accuracy_objects_dict)

# use XGBoost as stacked model.............................................
stacked_model_dict = dict()
# CatBoost (stacked),"iterations: 135 learning_rate: 0.25270630381399795 depth: 6,  l2_leaf_reg: 1.4654355982058695"
stacked_model_dict["CatBoost"] = CatB(iterations= 150, learning_rate = 0.25270630381399795, depth = 6, l2_leaf_reg = 1.4654355982058695)
# XGBoost (stacked),n_estimators: 380 max_depth: 15 subsample: 0.9829727315201009 
# colsample_bytree: 0.9821163261855509  gamma: 2.754057872914979 reg_alpha: 9.293502339665233  reg_lambda: 9.04947413810599
stacked_model_dict["XGBoost"] = XGBoost(n_estimators=380, max_depth = 15, subsample = 0.9829727315201009,
                                        colsample_bytree = 0.9821163261855509, gamma = 15, reg_alpha = 9.293502339665233,  reg_lambda = 9.04947413810599)
# LightBoost (stacked),"n_estimators: 185 learning_rate: 0.01596950334578271 max_depth: 19,  
# num_leaves: 30 min_child_samples: 84"
stacked_model_dict["LightBoost"] = LightB(n_estimators = 185, learning_rate = 0.01596950334578271, max_depth = 19, num_leaves = 30, min_child_samples = 84)

# AdaBoost (stacked),n_estimators: 200 learning_rate: 0.07286124900578073
stacked_model_dict["AdaBoost"] = AdaBoost(n_estimators= 200, learning_rate = 0.07286124900578073)
# GraBoost (stacked),"n_estimators: 146 learning_rate: 0.01596950334578271 
# max_depth: 15,  min_samples_leaf: 42 subsample: 0.5909124836035503"
stacked_model_dict["GraBoost"] = GBC(n_estimators= 146, learning_rate = 0.01596950334578271,
                                     max_depth = 15, min_samples_leaf = 42, subsample = 0.5909124836035503)

results_stacked = stacked_model_object_dictAND_accuracy_dict(meta_features_trainX, meta_features_trainY, meta_features_testX, 
                                           meta_features_testY, stacked_model_dict, accuracy_objects_dict)

print("Print and save final results")
for key in results_stacked.keys():
    result_dataframe[str(key) + str(" (stacked)")] = results_stacked[key]
df = pd.DataFrame.from_dict(result_dataframe, orient="index")
df.to_csv(path / "results_extracted_features.txt", index = True, sep = "\t")