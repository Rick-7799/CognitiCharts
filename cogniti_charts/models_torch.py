import torch, torch.nn as nn
class TorchCNN1D(nn.Module):
    def __init__(self, num_features, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(num_features, 48, 5, padding=2), nn.ReLU(), nn.BatchNorm1d(48),
            nn.Conv1d(48, 96, 5, padding=2), nn.ReLU(), nn.BatchNorm1d(96),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.25), nn.Linear(96,96), nn.ReLU(),
                                  nn.Dropout(0.15), nn.Linear(96,num_classes))
    def forward(self, x): x=x.transpose(1,2); x=self.net(x); return self.head(x)
