import torch
import torch.nn as nn


class ChessCNN(nn.Module):
    def __init__(self):
        super(ChessCNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=13, out_channels=32, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(64 * 4 * 4 + 5, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, board, additional_features):
        x = torch.relu(self.conv1(board))
        x = self.pool(torch.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)

        x = torch.cat([x, additional_features], dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
