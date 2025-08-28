from preprocessing.TON_IOT_multi_classification import *
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
# Train a machine learning model
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

import shap
import pandas as pd
import numpy as np
#shap.initjs()
from pathlib import Path

DATA_PATH = r'./data/raw/'
data_name = "ToN_IoT_train_test_network"

path = Path("results/multi/")
path.mkdir(parents=True, exist_ok=True)

X, y = data_load(DATA_PATH, data_name)
X_train, X_test, y_train, y_test = split_data_train_test(X, y)
# Split into train and test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

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
