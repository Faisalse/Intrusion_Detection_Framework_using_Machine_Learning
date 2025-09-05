from preprocessing.TON_IOT_multi_classification import *
from helper_functions import *
import shap
from sklearn.tree import DecisionTreeClassifier


DATA_PATH = r'./data/raw/'
data_name = "ToN_IoT_train_test_network"

X, y = data_load(DATA_PATH, data_name)
X_train, X_test, y_train, y_test = split_data_train_test(X, y)

model = DecisionTreeClassifier(max_depth=5, random_state=0)
model.fit(X_train, y_train)

explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test)



