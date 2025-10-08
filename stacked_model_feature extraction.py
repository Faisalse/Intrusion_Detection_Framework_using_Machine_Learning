from preprocessing.TON_IOT_multi_classification import *
from helper_functions import *
from accuracy.multi_accuracy import *
import pandas as pd
import torch
from denoising_autoencoder import *
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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
#models_object_dict["SVM"] = SVM()
models_object_dict["NB"] = NB(var_smoothing = 0.008562512157013471 )
models_object_dict["LR"] = LR(solver = "saga", penalty='none', C = 375)
models_object_dict["MLP"] = MLP(hidden_layer_sizes=(75, 57), alpha=0.08324593965363418, learning_rate_init = 0.01826431422398935, max_iter= 17)
#DT,Max_depth: 86 criterion: gini splitter: random
models_object_dict["DT"] = DTree(max_depth= 86, criterion="gini", splitter='random')

DATA_PATH = r'./data/raw/'
data_name = "ToN_IoT_train_test_network"

path = Path("results/multi/defaultHyperparameters/")
path.mkdir(parents=True, exist_ok=True)

X, y = data_load(DATA_PATH, data_name)
X_train, X_test, y_train, y_test = split_data_train_test(X, y)


X_tensor = torch.tensor(np.array(X_train), dtype=torch.float32)
X_tensor_t = torch.tensor(np.array(X_test), dtype=torch.float32)

input_dim = X_tensor.shape[1]
model = DenoisingAutoencoder(input_dim, hidden_dim=32, bottleneck_dim=10)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

train_loader = DataLoader(TensorDataset(X_tensor), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_tensor_t), batch_size=32, shuffle=False)

train_losses = []
test_losses = []

epochs = 50
for epoch in range(epochs):
    # ---- Training loss ----
    model.train()
    total_train_loss = 0
    for (batch,) in train_loader:
        noisy_batch = batch #+ 0.1 * torch.randn_like(batch)
        recon, _ = model(noisy_batch)
        loss = criterion(recon, batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    avg_train_loss = total_train_loss / len(train_loader)
    
    # ---- Validation/Test loss ----
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for (batch,) in test_loader:
            noisy_batch = batch + 0.1 * torch.randn_like(batch)
            recon, _ = model(noisy_batch)
            loss = criterion(recon, batch)
            total_test_loss += loss.item()
    avg_test_loss = total_test_loss / len(test_loader)
    
    # Save losses
    train_losses.append(avg_train_loss)
    test_losses.append(avg_test_loss)
    
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")




plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Denoising Autoencoder Loss")
plt.legend()
plt.show()



with torch.no_grad():
    _, X_train_features = model(X_tensor)
    _, X_test_features  = model(torch.tensor(np.array(X_test), dtype=torch.float32))


X_train = pd.DataFrame(X_train_features.numpy(),  columns= [i+1 for i in range(X_train_features.numpy().shape[1])])
X_test = pd.DataFrame(X_test_features.numpy(),  columns= [i+1 for i in range(X_test_features.numpy().shape[1])])
#X_train, X_test





meta_features_trainX, meta_features_trainY = k_fold_return_meta_features(X_train, y_train, models_object_dict, 
                                                                         accuracy_objects_dict, path)


meta_features_testX, meta_features_testY, result_dataframe = return_metafeatures_for_single_splits(X_train, y_train, X_test, 
                                                                                                       y_test, models_object_dict, 
                                                                                                       accuracy_objects_dict)

# use XGBoost as stacked model.............................................
stacked_model_dict = dict()
stacked_model_dict["XGBoost"] = XGBoost()
stacked_model_dict["CatBoost"] = CatB()
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