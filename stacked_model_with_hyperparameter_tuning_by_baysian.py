############################# IMPORT ALL LIBRARIES AND REQUIRED FILES ####################################

from preprocessing.TON_IOT_multi_classification import *
from helper_functions import *
from bayes_opt import BayesianOptimization
from accuracy.multi_accuracy import *
import time
import csv
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

############################ optimization files ---> baselines ########################################
from Optimization_files.opt_DT import *
from Optimization_files.opt_MLP import *
from Optimization_files.opt_lr import *
from Optimization_files.opt_nb import *
from Optimization_files.opt_svm import *
############################ optimization files ---> boosting algorithms ###############################
from Optimization_files.opt_CatBoost import *
from Optimization_files.opt_LightGBM import *
from Optimization_files.opt_XGBoost import *
from Optimization_files.opt_AdaBoost import *
from Optimization_files.opt_GraBoost import *

from functools import partial
from sklearn.model_selection import StratifiedKFold

models_object_dict = dict()
models_object_dict_time = dict()
############################# LOAD DATASET #############################################################
DATA_PATH = r'./data/raw/'
data_name = "ToN_IoT_train_test_network"

X, y = data_load(DATA_PATH, data_name)
X_train, X_test, y_train, y_test = split_data_train_test(X, y)

INTIAL_POINTS =5
N_ITERATIONS = 50
cv_strategy = StratifiedKFold(n_splits=5)

accuracy_objects_dict = dict()
accuracy_objects_dict["Accuracy"] = Acc()
accuracy_objects_dict["Precision"] = Precision()
accuracy_objects_dict["Recall"] = Recall()
accuracy_objects_dict["F1_score"] = F1_score()
path = Path("results/multi/optimalhyperparameter/")
path.mkdir(parents=True, exist_ok=True)


meta_features_trainX, meta_features_trainY = k_fold_return_meta_features(X_train, y_train, models_object_dict, 
                                                                            accuracy_objects_dict, path)

meta_features_testX, meta_features_testY, result_dataframe = return_metafeatures_for_single_splits(X_train, y_train, X_test, 
                                                                                                       y_test, models_object_dict, 
                                                                                                       accuracy_objects_dict)


################## Stacked models ##############################################################################

############################# FIND OPTIMAL HYPER-PARAMETER VALUES FOR GraBOOST #################################
start = time.time()
stacked_model_dict = dict()
opt_func = partial(
    optimize_gbc,
    X_train=meta_features_trainX,
    y_train=meta_features_trainY,
    X_valid=meta_features_testX,
    y_valid=meta_features_testY
)

optimizer = BayesianOptimization(
    f=opt_func,
    pbounds=gbcbounds,
    random_state=42,
    verbose=2
)

optimizer.maximize(init_points= INTIAL_POINTS, n_iter=N_ITERATIONS)
gbcBoost_optimal_hyperparameter_values = optimizer.max


n_estimators = round(gbcBoost_optimal_hyperparameter_values["params"]["n_estimators"])
max_depth = round(gbcBoost_optimal_hyperparameter_values["params"]["max_depth"])
learning_rate = gbcBoost_optimal_hyperparameter_values["params"]["learning_rate"]
subsample = gbcBoost_optimal_hyperparameter_values["params"]["subsample"]
min_samples_leaf = round(gbcBoost_optimal_hyperparameter_values["params"]["min_samples_leaf"])

stacked_model_dict["GraBoost (stacked)"] = GBC(n_estimators = n_estimators, learning_rate = learning_rate, max_depth = max_depth, 
                                     min_samples_leaf  = min_samples_leaf, subsample = subsample)
models_object_dict_time["GraBoost (stacked)"] = time.time() - start
############################# FIND OPTIMAL HYPER-PARAMETER VALUES FOR AdaBOOST #################################
start = time.time()
opt_func = partial(
    optimize_adaboost,
    X_train=meta_features_trainX,
    y_train=meta_features_trainY,
    X_valid=meta_features_testX,
    y_valid=meta_features_testY
)

optimizer = BayesianOptimization(
    f=opt_func,
    pbounds=adaboost_search_space,
    random_state=42,
    verbose=2
)

optimizer.maximize(init_points= INTIAL_POINTS, n_iter=20)
AdaBoost_optimal_hyperparameter_values = optimizer.max

learning_rate = AdaBoost_optimal_hyperparameter_values["params"]["learning_rate"]
n_estimators = round(AdaBoost_optimal_hyperparameter_values["params"]["n_estimators"])
stacked_model_dict["AdaBoost (stacked)"] = AdaBoost(n_estimators = n_estimators, learning_rate = learning_rate)
models_object_dict_time["AdaBoost (stacked)"] = time.time() - start
############################# FIND OPTIMAL HYPER-PARAMETER VALUES FOR CatBOOST #################################
start = time.time()
opt_func = partial(
    optimize_catb,
    X_train=meta_features_trainX,
    y_train=meta_features_trainY,
    X_valid=meta_features_testX,
    y_valid=meta_features_testY
)

optimizer = BayesianOptimization(
    f=opt_func,
    pbounds=catbounds,
    random_state=42,
    verbose=2
)

optimizer.maximize(init_points= INTIAL_POINTS, n_iter=N_ITERATIONS)
CatBoost_optimal_hyperparameter_values = optimizer.max

depth = round(CatBoost_optimal_hyperparameter_values["params"]["depth"])
iterations = round(CatBoost_optimal_hyperparameter_values["params"]["iterations"])
learning_rate = CatBoost_optimal_hyperparameter_values["params"]["learning_rate"]
l2_leaf_reg = CatBoost_optimal_hyperparameter_values["params"]["l2_leaf_reg"]

stacked_model_dict["CatBoost (stacked)"] = CatB(iterations = iterations, learning_rate = learning_rate, depth = depth, l2_leaf_reg = l2_leaf_reg)
models_object_dict_time["CatBoost (stacked)"] = time.time() - start
############################# FIND OPTIMAL HYPER-PARAMETER VALUES FOR LIGHTGBM #################################
start = time.time()
opt_func = partial(
    optimize_lightb,
    X_train=meta_features_trainX,
    y_train=meta_features_trainY,
    X_valid=meta_features_testX,
    y_valid=meta_features_testY
)

optimizer = BayesianOptimization(
    f=opt_func,
    pbounds=lightbounds,
    random_state=42,
    verbose=2
)

optimizer.maximize(init_points= INTIAL_POINTS, n_iter=N_ITERATIONS)
lightBoost_optimal_hyperparameter_values = optimizer.max

n_estimators = round(lightBoost_optimal_hyperparameter_values['params']["n_estimators"])
learning_rate = lightBoost_optimal_hyperparameter_values['params']["learning_rate"]
max_depth = round(lightBoost_optimal_hyperparameter_values['params']["max_depth"])
num_leaves = round(lightBoost_optimal_hyperparameter_values['params']["num_leaves"])
min_child_samples = round(lightBoost_optimal_hyperparameter_values['params']["min_child_samples"])

stacked_model_dict["LightBoost (stacked)"] = LightB(n_estimators = n_estimators, learning_rate = learning_rate, max_depth = max_depth, 
                 num_leaves = num_leaves, min_child_samples = min_child_samples)
models_object_dict_time["LightBoost (stacked)"] = time.time() - start
############################# FIND OPTIMAL HYPER-PARAMETER VALUES FOR XGBoost #################################
start = time.time()
opt_func = partial(
    optimize_xgb,
    X_train=meta_features_trainX,
    y_train=meta_features_trainY,
    X_valid=meta_features_testX,
    y_valid=meta_features_testY
)
optimizer = BayesianOptimization(
    f=opt_func,
    pbounds=xgbounds,
    random_state=42,
    verbose=2
)

optimizer.maximize(init_points = INTIAL_POINTS, n_iter = N_ITERATIONS)
xgBoost_optimal_hyperparameter_values = optimizer.max

n_estimators = round(xgBoost_optimal_hyperparameter_values["params"]["n_estimators"])
max_depth = round(xgBoost_optimal_hyperparameter_values["params"]["max_depth"])
learning_rate = xgBoost_optimal_hyperparameter_values["params"]["learning_rate"]
subsample = xgBoost_optimal_hyperparameter_values["params"]["subsample"]
colsample_bytree = xgBoost_optimal_hyperparameter_values["params"]["colsample_bytree"]
gamma = xgBoost_optimal_hyperparameter_values["params"]["gamma"]
reg_alpha = xgBoost_optimal_hyperparameter_values["params"]["reg_alpha"]
reg_lambda = xgBoost_optimal_hyperparameter_values["params"]["reg_lambda"]


stacked_model_dict["XGBoost (stacked)"] = XGBoost(n_estimators = n_estimators, max_depth = max_depth, learning_rate = learning_rate, subsample = subsample, 
                                        colsample_bytree = colsample_bytree, 
                                        gamma = gamma, reg_alpha = reg_alpha, reg_lambda = reg_lambda)
models_object_dict_time["XGBoost (stacked)"] = time.time() - start


results_stacked = stacked_model_object_dictAND_accuracy_dict(meta_features_trainX, meta_features_trainY, meta_features_testX, 
                                           meta_features_testY, stacked_model_dict, accuracy_objects_dict)


print("Print and save final results")
for key in results_stacked.keys():
    result_dataframe[key] = results_stacked[key]
for key in models_object_dict_time.keys():
    result_dataframe[key]["tuning_time"] = models_object_dict_time[key]

df = pd.DataFrame.from_dict(result_dataframe, orient="index")
df.to_csv(path / "optimalHyperparameters.txt", index = True, sep = "\t")


merged = {**stacked_model_dict, **models_object_dict}
merged2 = dict() 
for key in merged.keys():
    merged2[key] = merged[key].use_hyperparameter_value()
with open(path /"optimal_hyperparameterValue.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Model", "Optimal Hyperparameter Values"])  # header
    for key, value in merged2.items():
        writer.writerow([key, value])


