import torch
import torch.nn as nn

class VModel(nn.Module):
    def __init__(self, n_frame, n_action):
        super(VModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=n_frame, out_channels=16,
                               kernel_size=8, stride=4, padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32,
                              kernel_size=4, stride=2)
        self.fc1 = nn.Linear(32 * 81, 256)
        self.fc2 = nn.Linear(256, n_action)
        self.act = nn.ReLU()
        
    def forward(self, image):
        x = self.act(self.conv1(image))
        x = self.act(self.conv2(x))
        x = x.view(-1, 32 * 81)
        x = self.act(self.fc1(x))
        return self.fc2(x)
