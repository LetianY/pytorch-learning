import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import robust_loss_pytorch.general
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def Normalize(data):
    data_mean = torch.mean(data, dim=0)
    data_max = torch.max(data, dim=0)[0]
    data_min = torch.min(data, dim=0)[0]
    data = (data-data_mean)/(data_max-data_min)
    return data

def plot_regression(pred_y, actual_y):
    x_plot = np.linspace(0, 1, 100)
    plt.figure(0, figsize=(4,4))
    plt.scatter(actual_y, pred_y)
    plt.plot(x_plot, x_plot, color='r')
    # plt.xlim([-4, 4])
    # plt.ylim([-4, 4])
  
class RegressionModel(torch.nn.Module): 
    # A simple linear regression module.
    def __init__(self): 
        super(RegressionModel, self).__init__() 
        self.linear = torch.nn.Linear(len(features), 1)
    def forward(self, x): 
        return self.linear(x[:,None]).squeeze(1)
  
# Data Processing
train_X = torch.Tensor(train_ds[features].values)
train_y = torch.Tensor(np.asarray(train_ds[y_label].values).reshape(-1, 1))

test_X = torch.Tensor(test_ds[features].values)
test_y = torch.Tensor(np.asarray(test_ds[y_label].values).reshape(-1, 1))

train_X_normal = Normalize(train_X)
test_X_normal = Normalize(test_X)

# Construct Training Data
batch_size = 256
epochs = 1000
trainset = TensorDataset(train_X_normal, train_y)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

# Fit a linear regression, and the parameters of an adaptive loss.
regression = RegressionModel() 
adaptive = robust_loss_pytorch.adaptive.AdaptiveLossFunction(num_dims = 1, float_dtype=np.float32, device='cpu')
params = list(regression.parameters()) + list(adaptive.parameters())
optimizer = torch.optim.Adam(params, lr = 0.01) 

for epoch in range(epochs):
    for batch_feature, batch_target in trainloader:
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize:
        batch_pred = regression(batch_feature)
        loss = torch.mean(adaptive.lossfun((batch_pred - batch_target)))
        loss.backward()
        optimizer.step()
        
    # print statistics
    alpha = adaptive.alpha()[0, 0]
    scale = adaptive.scale()[0, 0]
    print(f"Epoch {epoch}/{epochs}: Loss: {loss}, alpha: {alpha:.3f}, scale: {scale:.3f}")
    print('----------------------------------------------------------')
print('Finished Training')

# Save Model
PATH = './robust_regression.pth'
torch.save(regression.state_dict(), PATH)

# Construct Testing Dataset
testset = TensorDataset(test_X_normal, test_y)
testloader = torch.utils.data.DataLoader(testset, batch_size=test_y.shape[0], shuffle=False, num_workers=2)

# Prediction
with torch.no_grad():
    for X_test, y_test in testloader:
        pred_test = regression(X_test)

# Plot
plot_regression(pred_test, y_test)

      
