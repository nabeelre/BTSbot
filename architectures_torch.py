import torch
import torch.nn as nn


class SwinV2_t(nn.Module):
    def __init__(self, config):
        super(SwinV2_t, self).__init__()
        self.swin = torch.hub.load(
            "pytorch/vision", "swin_v2_t", weights="IMAGENET1K_V1", progress=False
        )
        self.fc = nn.Sequential(
            nn.Linear(1000, config['fc1_neurons']),
            nn.Linear(config['fc1_neurons'], config['fc2_neurons']),
            nn.ReLU(True),
            nn.Dropout(config['dropout1']),
            nn.Linear(config['fc2_neurons'], 1),
            nn.Sigmoid()
        )

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        outputs = self.swin(input_data)
        preds = self.fc(outputs)
        return preds.view(-1)


class SwinV2_s(nn.Module):
    def __init__(self, config):
        super(SwinV2_s, self).__init__()
        self.swin = torch.hub.load(
            "pytorch/vision", "swin_v2_s", weights="IMAGENET1K_V1", progress=False
        )
        self.fc = nn.Sequential(
            nn.Linear(1000, config['fc1_neurons']),
            nn.Linear(config['fc1_neurons'], config['fc2_neurons']),
            nn.ReLU(True),
            nn.Dropout(config['dropout1']),
            nn.Linear(config['fc2_neurons'], 1),
            nn.Sigmoid()
        )

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        outputs = self.swin(input_data)
        preds = self.fc(outputs)
        return preds.view(-1)
