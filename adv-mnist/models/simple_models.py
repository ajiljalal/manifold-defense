import torch as ch
import numpy as np
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as snorm
from torch import nn

class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.fc = nn.Linear(3136, 1024)
        self.final_fc = nn.Linear(1024, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.shape[0], -1)
        out = F.relu(self.fc(out))
        return self.final_fc(out)

class SmallSimpleClassifier(nn.Module):
    def __init__(self, scale=10):
        super(SmallSimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(128, 400*scale)
        self.bn1 = nn.BatchNorm1d(400*scale)
        self.fc2 = nn.Linear(400*scale, 200*scale)
        self.bn2 = nn.BatchNorm1d(200*scale)
        self.fc3 = nn.Linear(200*scale, 100*scale)
        self.bn3 = nn.BatchNorm1d(100*scale)
        self.fc4 = nn.Linear(100*scale, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        return self.fc4(x)
