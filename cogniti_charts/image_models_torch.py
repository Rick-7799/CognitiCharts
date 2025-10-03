import torch, torch.nn as nn
class TorchImageCNN(nn.Module):
    def __init__(self,num_classes=3):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1))
        )
        self.head=nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128,128), nn.ReLU(), nn.Dropout(0.15), nn.Linear(128,num_classes))
    def forward(self,x): x=self.features(x); return self.head(x)
