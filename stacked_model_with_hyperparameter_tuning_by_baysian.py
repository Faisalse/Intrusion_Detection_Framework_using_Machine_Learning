############################# IMPORT ALL LIBRARIES AND REQUIRED FILES ####################################

from preprocessing.TON_IOT_multi_classification import *
from helper_functions import *
from bayes_opt import BayesianOptimization
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

from functools import partial
from sklearn.model_selection import StratifiedKFold

models_object_dict = dict()
############################# LOAD DATASET #############################################################
DATA_PATH = r'./data/raw/'
data_name = "ToN_IoT_train_test_network"

X, y = data_load(DATA_PATH, data_name)
X_train, X_test, y_train, y_test = split_data_train_test(X, y)

INTIAL_POINTS = 1
N_ITERATIONS = 1
cv_strategy = StratifiedKFold(n_splits=5)

############################# FIND OPTIMAL HYPER-PARAMETER VALUES FOR SVM ###############################
"""
opt_func = partial(optimize_svm, X= X_train, y=y_train, cv=cv_strategy)
optimizer = BayesianOptimization(
    f=opt_func,
    pbounds=svm_bounds,
    random_state=42,
    verbose=2
)

optimizer.maximize(init_points = INTIAL_POINTS, n_iter = N_ITERATIONS)
DT_optimal_hyperparameter_values = optimizer.max

criterion = criterion_map[round(DT_optimal_hyperparameter_values["params"]["criterion"] )]
splitter = splitter_map[round(DT_optimal_hyperparameter_values["params"]['splitter'])]

models_object_dict["DT"] = DTree(criterion = criterion, 
                                 max_depth = DT_optimal_hyperparameter_values["params"]['max_depth'], 
                                 splitter = splitter)

"""
############################# FIND OPTIMAL HYPER-PARAMETER VALUES FOR DT #################################
opt_func = partial(optimize_dtree, X= X_train, y=y_train, cv=cv_strategy)
optimizer = BayesianOptimization(
    f=opt_func,
    pbounds=dtbounds,
    random_state=42,
    verbose=2
)

optimizer.maximize(init_points = INTIAL_POINTS, n_iter = N_ITERATIONS)
DT_optimal_hyperparameter_values = optimizer.max

criterion = criterion_map[round(DT_optimal_hyperparameter_values["params"]["criterion"] )]
splitter = splitter_map[round(DT_optimal_hyperparameter_values["params"]['splitter'])]

models_object_dict["DT"] = DTree(criterion = criterion, 
                                 max_depth = DT_optimal_hyperparameter_values["params"]['max_depth'], 
                                 splitter = splitter)

"""
############################# FIND OPTIMAL HYPER-PARAMETER VALUES FOR LR #################################
opt_func = partial(optimize_lr, X = X_train, y = y_train, cv = cv_strategy)
optimizer = BayesianOptimization(
    f=opt_func,
    pbounds=lrpbounds,
    random_state=42,
    verbose=2
)

optimizer.maximize(init_points= INTIAL_POINTS, n_iter=N_ITERATIONS)
lr_optimal_hyperparameter_values = optimizer.max

solver = solver_map[round(lr_optimal_hyperparameter_values["params"]["solver"])]
penalty = penalty_map[round(lr_optimal_hyperparameter_values["params"]["penalty"])]
C = round(lr_optimal_hyperparameter_values["params"]["C"])
models_object_dict["LR"] = LR(solver = solver, penalty = penalty, C = C)

############################# FIND OPTIMAL HYPER-PARAMETER VALUES FOR NB #################################
opt_func = partial(optimize_nb, X = X_train, y = y_train, cv = cv_strategy)
optimizer = BayesianOptimization(
    f=opt_func,
    pbounds=nb_pbounds,
    random_state=42,
    verbose=2
)

optimizer.maximize(init_points= INTIAL_POINTS, n_iter=N_ITERATIONS)
nb_optimal_hyperparameter_values = optimizer.max
models_object_dict["NB"] = NB(var_smoothing = nb_optimal_hyperparameter_values['params']['var_smoothing'])

############################# FIND OPTIMAL HYPER-PARAMETER VALUES FOR MLP #################################
opt_func = partial(optimize_mlp, X = X_train, y = y_train, cv = cv_strategy)
optimizer = BayesianOptimization(
    f=opt_func,
    pbounds=MLPpbounds,
    random_state=42,
    verbose=2
)

optimizer.maximize(init_points= INTIAL_POINTS, n_iter=N_ITERATIONS)
mLP_optimal_hyperparameter_values = optimizer.max

hidden1 = round(mLP_optimal_hyperparameter_values["params"]["units1"])
hidden2 = round(mLP_optimal_hyperparameter_values["params"]["units2"])
alpha = mLP_optimal_hyperparameter_values["params"]["alpha"]
learning_rate = lr_map[round(mLP_optimal_hyperparameter_values["params"]["learning_rate"])]
learning_rate_init = mLP_optimal_hyperparameter_values["params"]["learning_rate_init"]
max_iter = round(mLP_optimal_hyperparameter_values["params"]["max_iter"])


models_object_dict["MLP"] = MLP(hidden_layer_sizes = (hidden1, hidden2), alpha = alpha,
                 learning_rate = learning_rate, learning_rate_init = learning_rate_init, max_iter = max_iter)

"""




############################# ACCURACY MEASURES #################################
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

############################# FIND OPTIMAL HYPER-PARAMETER VALUES FOR AdaBOOST #################################
stacked_model_dict = dict()
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

optimizer.maximize(init_points= INTIAL_POINTS, n_iter=N_ITERATIONS)
AdaBoost_optimal_hyperparameter_values = optimizer.max


learning_rate = AdaBoost_optimal_hyperparameter_values["params"]["depth"]
n_estimators = round(AdaBoost_optimal_hyperparameter_values["params"]["n_estimators"])


stacked_model_dict["AdaBoost"] = AdaBoost(learning_rate = learning_rate, n_estimators = n_estimators)



############################# FIND OPTIMAL HYPER-PARAMETER VALUES FOR CatBOOST #################################
stacked_model_dict = dict()
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

stacked_model_dict["CatB"] = CatB(iterations = iterations, learning_rate = learning_rate, depth = depth, l2_leaf_reg = l2_leaf_reg)

############################# FIND OPTIMAL HYPER-PARAMETER VALUES FOR LIGHTGBM #################################
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

stacked_model_dict["LightB"] = LightB(n_estimators = n_estimators, learning_rate = learning_rate, max_depth = max_depth, 
                 num_leaves = num_leaves, min_child_samples = min_child_samples)

############################# FIND OPTIMAL HYPER-PARAMETER VALUES FOR XGBoost #################################
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


stacked_model_dict["XGBoost"] = XGBoost(n_estimators = n_estimators, max_depth = max_depth, learning_rate = learning_rate, subsample = subsample, 
                                        colsample_bytree = colsample_bytree, 
                                        gamma = gamma, reg_alpha = reg_alpha, reg_lambda = reg_lambda)

results_stacked = stacked_model_object_dictAND_accuracy_dict(meta_features_trainX, meta_features_trainY, meta_features_testX, 
                                           meta_features_testY, stacked_model_dict, accuracy_objects_dict)


print("Print and save final results")
for key in results_stacked.keys():
    result_dataframe[key] = results_stacked[key]
df = pd.DataFrame.from_dict(result_dataframe, orient="index")
df.to_csv(path / "optimalHyperparameters.txt", index = True, sep = "\t")
