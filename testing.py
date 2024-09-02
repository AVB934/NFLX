import torch
import torch.nn as nn

class LinearRegression(nn.Module):
    def __init__(self, input_features):
        super(LinearRegression, self).__init__()
        self.layer = nn.Linear(input_features, 1)

    def forward(self, X):
        return self.layer(X)


input_features = 4
model = LinearRegression(input_features)
model.load_state_dict(torch.load('nflx_model.pth'))

