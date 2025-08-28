
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from pathlib import Path
import random
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def apply_random_scaling_df_global(df, ratio=0.1, scale=0.1, seed=43):
    """
    Randomly selects a ratio of values in the DataFrame and scales them by a given factor.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        ratio (float): Fraction of total elements to scale (default is 0.1 for 10%).
        scale (float): Multiplicative scale factor (default is 0.1).
        seed (int, optional): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Modified DataFrame with scaled values.
    """
    if seed is not None:
        np.random.seed(seed)

    df_copy = df.copy()
    total_elements = df_copy.size
    num_to_scale = int(total_elements * ratio)

    # Generate random (row, column) index pairs
    row_indices = np.random.randint(0, df_copy.shape[0], num_to_scale)
    col_indices = np.random.randint(0, df_copy.shape[1], num_to_scale)

    for row, col in zip(row_indices, col_indices):
        col_name = df_copy.columns[col]
        df_copy.iat[row, col] *= scale

    return df_copy


def apply_random_scaling_df_local(df, ratio=0.1, seed=43):
    
    if seed is not None:
        np.random.seed(seed)
    cols = df.columns
    df_copy = df.copy()
    continuous_cols = df_copy.select_dtypes(include=['float']).columns.tolist()

    selected_col = continuous_cols[random.choice([i for i in range(len(continuous_cols))])]
    df_copy[selected_col] = df_copy[selected_col] * ratio
    return df_copy


def k_fold_return_meta_features(X_train, y_train, models_object_dict, accuracy_objects_dict, path, n_splits = 5, 
                                random_state = 42, attack = False):
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    

    meta_features = dict()
    for key in models_object_dict.keys():
        meta_features[key] = list()
    meta_y = list()

    # We are using out of fold strategy to avoid data leakage issue............
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    result_dataframe = dict()
    for fold, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
        for key in models_object_dict.keys():
            print(f"*********************** {key} ***********************")
            models_object_dict[key].fit(X_train[train_index], y_train[train_index])

            y_predict = models_object_dict[key].predict(X_train[test_index])
            y_predict_prob = models_object_dict[key].predict_proba(X_train[test_index])

            column_names = [i for i in range(y_predict_prob.shape[1])]
            temp_df = pd.DataFrame(y_predict_prob, columns= column_names)
            meta_features[key].append(temp_df)
            
            # print accuracy values on out of fold
            temp = dict()
            for acc_key, acc_object in accuracy_objects_dict.items():
                acc_object.compute(y_predict, y_train[test_index])
                temp[acc_key] = round(acc_object.result()[1], 4)
                print(acc_object.result())
                
            result_dataframe[str(key) +"_fold_"+str(fold)] = temp

        meta_y.append(y_train[test_index])   
    
    if attack == False:
        result_dataframe = pd.DataFrame.from_dict(result_dataframe, orient='index')
        path = path / "results_with_folding.csv"
        result_dataframe.to_csv(path, sep = "\t")

    meta_features_df = pd.DataFrame()
    for key, value in meta_features.items():
        temp = pd.concat(value, axis=0, ignore_index=True)
        meta_features_df = pd.concat([meta_features_df, temp], axis= 1, ignore_index=True)

    # make meta features frame.........................................................
    new_column_names = [i for i in   range(  meta_features_df.shape[1] )]  
    meta_features_df.columns = new_column_names
    meta_features_y = [item   for sublist in meta_y for item in list(sublist)]

    return meta_features_df, meta_features_y



def return_metafeatures_for_single_splits(X_train, y_train, X_test, y_test, models_object_dict, accuracy_objects_dict):
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()

    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    result_dataframe = dict()

    meta_features = dict()
    for key in models_object_dict.keys():
        meta_features[key] = list()
    # results with full data
    for key in models_object_dict.keys():
        print(f"*********************** Results with full data: {key} ***********************")
        models_object_dict[key].fit(X_train, y_train)

        
        y_predict = models_object_dict[key].predict(X_test)
        y_predict_prob = models_object_dict[key].predict_proba(X_test)

        column_names = [i for i in range(y_predict_prob.shape[1])]
        temp_df = pd.DataFrame(y_predict_prob, columns= column_names)
        meta_features[key].append(temp_df)
            
            # print accuracy values on out of fold
        temp = dict()
        for acc_key, acc_object in accuracy_objects_dict.items():
            acc_object.compute(y_predict, y_test)
            temp[acc_key] = round(acc_object.result()[1], 4)
            print(acc_object.result())

        result_dataframe[str(key)] = temp
    
    meta_features_df = pd.DataFrame()
    for key, value in meta_features.items():
        meta_features_df = pd.concat([meta_features_df, value[0]], axis= 1, ignore_index=True)

    # make meta features frame.........................................................
    new_column_names = [i for i in   range(  meta_features_df.shape[1] )]  
    meta_features_df.columns = new_column_names
    return meta_features_df, y_test, result_dataframe

def stacked_model_object_dictAND_accuracy_dict(X_train, y_train, X_test, y_test, models_object_dict, accuracy_objects_dict):

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    
    
    result_dataframe = dict()
    for key in models_object_dict.keys():
        print(f"*********************** Results with full data: {key} ***********************")
        models_object_dict[key].fit(X_train, y_train)
        

        y_predict = models_object_dict[key].predict(X_test)
        y_predict_prob = models_object_dict[key].predict_proba(X_test)

        X_test
          
        # print accuracy values on out of fold
        temp = dict()
        for acc_key, acc_object in accuracy_objects_dict.items():
            acc_object.compute(y_predict, y_test)
            print(acc_object.result())
            temp[acc_key] = round(acc_object.result()[1], 4)
        result_dataframe[key] = temp
        

    return result_dataframe