import torch
import torch.nn as nn


class SwinV2(nn.Module):
    def __init__(self, config):
        super(SwinV2, self).__init__()
        model_kind = config.get("model_kind", "swin_v2_t")
        model_weights = config.get("model_weights", "IMAGENET1K_V1")

        self.swin = torch.hub.load(
            "pytorch/vision", model_kind, weights=model_weights, progress=False
        )
        self.swin.head = nn.Sequential(
            nn.Linear(self.swin.head.in_features, config['fc1_neurons']),
            nn.Linear(config['fc1_neurons'], config['fc2_neurons']),
            nn.ReLU(True),
            nn.Dropout(config['dropout1']),
            nn.Linear(config['fc2_neurons'], 1),
        )

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        return self.swin(input_data)


class SwinV2_s(nn.Module):
    def __init__(self, config):
        super(SwinV2_s, self).__init__()
        self.swin = torch.hub.load(
            "pytorch/vision", "swin_v2_s", weights="IMAGENET1K_V1", progress=False
        )
        self.swin.head = nn.Sequential(
            nn.Linear(self.swin.head.in_features, config['fc1_neurons']),
            nn.Linear(config['fc1_neurons'], config['fc2_neurons']),
            nn.ReLU(True),
            nn.Dropout(config['dropout1']),
            nn.Linear(config['fc2_neurons'], 1),
        )

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        return self.swin(input_data)
