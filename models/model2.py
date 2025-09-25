import torch
import torch.nn as nn
import torch.nn.functional as F

class MDL(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64 , 10)

    def forward(self,x):
        x = x.view(-1,28^2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        #not done editing this additional model yet. I want to tweak it and see how
        #i can change it to perhaps be more accurate or predict better.
        return x