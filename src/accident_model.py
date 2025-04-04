import torch.nn as nn


class AccidentPredictionModel(nn.Module):
    def __init__(self, input_size):
        super(AccidentPredictionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 36)
        self.fc2 = nn.Linear(36, 36)
        self.fc3 = nn.Linear(36, 18)
        self.fc4 = nn.Linear(18, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x
