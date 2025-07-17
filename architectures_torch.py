import torch
import torch.nn as nn
import timm
import re


def get_model_image_size(model_kind: str) -> int:
    """Extract the image size from model name.
    For example:
    - maxvit_base_tf_224.in1k -> 224
    - maxvit_large_tf_384.in1k -> 384
    - swin_v2_t -> 256 (default for SwinV2)
    """
    if 'maxvit' in model_kind.lower():
        # Extract size from model name using regex
        match = re.search(r'_(\d+)\.', model_kind)
        if match:
            return int(match.group(1))
    return 224


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
            nn.GELU(),
            nn.Linear(config['fc1_neurons'], config['fc2_neurons']),
            nn.GELU(),
            nn.Dropout(config['dropout1']),
            nn.Linear(config['fc2_neurons'], 1),
        )

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        return self.swin(input_data)


class mm_SwinV2(nn.Module):
    def __init__(self, config):
        super(mm_SwinV2, self).__init__()
        swin_kind = config.get("swin_kind", "swin_v2_t")
        swin_weights = config.get("swin_weights", "IMAGENET1K_V1")
        num_metadata_features = len(config.get("metadata_cols", []))

        # Image branch (SwinV2)
        self.swin_backbone = torch.hub.load(
            "pytorch/vision", swin_kind, weights=swin_weights, progress=False
        )
        self.swin_feature_dim = self.swin_backbone.head.in_features
        self.swin_backbone.head = nn.Identity()

        # Metadata branch
        self.metadata_branch = nn.Sequential(
            nn.BatchNorm1d(num_metadata_features),
            nn.Linear(num_metadata_features, config['meta_fc1_neurons']),
            nn.GELU(),
            nn.Dropout(config['meta_dropout']),
            nn.Linear(config['meta_fc1_neurons'], config['meta_fc2_neurons']),
            nn.GELU()
        )

        # Combined branch
        combined_input_features = self.swin_feature_dim + config['meta_fc2_neurons']
        self.combined_head = nn.Sequential(
            nn.Linear(combined_input_features, config['comb_fc1_neurons']),
            nn.GELU(),
            nn.Linear(config['comb_fc1_neurons'], config['comb_fc2_neurons']),
            nn.GELU(),
            nn.Dropout(config['comb_dropout']),
            nn.Linear(config['comb_fc2_neurons'], 1)
        )

    def forward(self, image_input: torch.Tensor, metadata_input: torch.Tensor) -> torch.Tensor:
        image_features = self.swin_backbone(image_input)
        meta_features = self.metadata_branch(metadata_input)
        combined_features = torch.cat((image_features, meta_features), dim=1)
        logits = self.combined_head(combined_features)

        return logits


class MaxViT(nn.Module):
    def __init__(self, config):
        super(MaxViT, self).__init__()
        model_kind = config.get("model_kind", "maxvit_tiny_rw_224.sw_in1k")
        self.image_size = get_model_image_size(model_kind)

        self.maxvit = timm.create_model(model_kind, pretrained=config.get('pretrained', True))
        self.maxvit.head = nn.Sequential(
            self.maxvit.head.global_pool,
            nn.Linear(self.maxvit.head.in_features, config['fc1_neurons']),
            nn.GELU(),
            nn.Linear(config['fc1_neurons'], config['fc2_neurons']),
            nn.GELU(),
            nn.Dropout(config['dropout1']),
            nn.Linear(config['fc2_neurons'], 1),
        )

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        # Resize input to expected size if needed
        if input_data.shape[-1] != self.image_size or input_data.shape[-2] != self.image_size:
            input_data = torch.nn.functional.interpolate(
                input_data,
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            )
        return self.maxvit(input_data)


class mm_MaxViT(nn.Module):
    def __init__(self, config):
        super(mm_MaxViT, self).__init__()
        model_kind = config.get("model_kind", "maxvit_tiny_rw_224.sw_in1k")
        self.image_size = get_model_image_size(model_kind)
        num_metadata_features = len(config.get("metadata_cols", []))

        # Image branch (MaxViT)
        self.maxvit_backbone = timm.create_model(model_kind,
                                                 pretrained=config.get('pretrained', True))
        self.maxvit_feature_dim = self.maxvit_backbone.head.in_features
        self.maxvit_backbone.head = self.maxvit_backbone.head.global_pool

        # Metadata branch
        self.metadata_branch = nn.Sequential(
            nn.BatchNorm1d(num_metadata_features),
            nn.Linear(num_metadata_features, config['meta_fc1_neurons']),
            nn.GELU(),
            nn.Dropout(config['meta_dropout']),
            nn.Linear(config['meta_fc1_neurons'], config['meta_fc2_neurons']),
            nn.GELU()
        )

        # Combined branch
        combined_input_features = self.maxvit_feature_dim + config['meta_fc2_neurons']
        self.combined_head = nn.Sequential(
            nn.Linear(combined_input_features, config['comb_fc1_neurons']),
            nn.GELU(),
            nn.Linear(config['comb_fc1_neurons'], config['comb_fc2_neurons']),
            nn.GELU(),
            nn.Dropout(config['comb_dropout']),
            nn.Linear(config['comb_fc2_neurons'], 1)
        )

    def forward(self, image_input: torch.Tensor, metadata_input: torch.Tensor) -> torch.Tensor:
        # Resize input to expected size if needed
        if image_input.shape[-1] != self.image_size or image_input.shape[-2] != self.image_size:
            image_input = torch.nn.functional.interpolate(
                image_input,
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            )
        image_features = self.maxvit_backbone(image_input)
        meta_features = self.metadata_branch(metadata_input)
        combined_features = torch.cat((image_features, meta_features), dim=1)
        logits = self.combined_head(combined_features)
        return logits


class ConvNeXt(nn.Module):
    def __init__(self, config):
        super(ConvNeXt, self).__init__()
        model_kind = config.get("model_kind", "convnext_nano.d1h_in1k")
        self.convnext = timm.create_model(model_kind, pretrained=config.get('pretrained', True))
        self.convnext.head = nn.Sequential(
            self.convnext.head.global_pool,
            self.convnext.head.norm,
            self.convnext.head.flatten,
            nn.Linear(self.convnext.head.in_features, config['fc1_neurons']),
            nn.GELU(),
            nn.Linear(config['fc1_neurons'], config['fc2_neurons']),
            nn.GELU(),
            nn.Dropout(config['dropout1']),
            nn.Linear(config['fc2_neurons'], 1),
        )

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        return self.convnext(input_data)


class mm_ConvNeXt(nn.Module):
    def __init__(self, config):
        super(mm_ConvNeXt, self).__init__()
        model_kind = config.get("model_kind", "convnext_nano.d1h_in1k")
        num_metadata_features = len(config.get("metadata_cols", []))

        # Image branch (ConvNeXt)
        self.convnext_backbone = timm.create_model(model_kind,
                                                   pretrained=config.get('pretrained', True))
        self.convnext_feature_dim = self.convnext_backbone.head.in_features
        # Add global pool and norm to head when using larger legacy survey images
        if "LS" in config['train_data_version']:
            self.convnext_backbone.head = nn.Sequential(
                self.convnext_backbone.head.global_pool,
                self.convnext_backbone.head.norm,
                self.convnext_backbone.head.flatten
            )
        else:
            self.convnext_backbone.head = self.convnext_backbone.head.flatten

        # Metadata branch
        self.metadata_branch = nn.Sequential(
            nn.BatchNorm1d(num_metadata_features),
            nn.Linear(num_metadata_features, config['meta_fc1_neurons']),
            nn.GELU(),
            nn.Dropout(config['meta_dropout']),
            nn.Linear(config['meta_fc1_neurons'], config['meta_fc2_neurons']),
            nn.GELU()
        )

        # Combined branch
        combined_input_features = self.convnext_feature_dim + config['meta_fc2_neurons']
        self.combined_head = nn.Sequential(
            nn.Linear(combined_input_features, config['comb_fc1_neurons']),
            nn.GELU(),
            nn.Linear(config['comb_fc1_neurons'], config['comb_fc2_neurons']),
            nn.GELU(),
            nn.Dropout(config['comb_dropout']),
            nn.Linear(config['comb_fc2_neurons'], 1)
        )

    def forward(self, image_input: torch.Tensor, metadata_input: torch.Tensor) -> torch.Tensor:
        image_features = self.convnext_backbone(image_input)
        meta_features = self.metadata_branch(metadata_input)
        combined_features = torch.cat((image_features, meta_features), dim=1)
        logits = self.combined_head(combined_features)
        return logits


class mm_ResNet(nn.Module):
    def __init__(self, config):
        super(mm_ResNet, self).__init__()
        model_kind = config.get("model_kind", "resnet18.a1_in1k")
        num_metadata_features = len(config.get("metadata_cols", []))

        # Image branch (ResNet)
        self.resnet_backbone = timm.create_model(model_kind,
                                                 pretrained=config.get('pretrained', True))
        self.resnet_feature_dim = self.resnet_backbone.fc.in_features
        # Add global pool and norm to head when using larger legacy survey images
        self.resnet_backbone.fc = nn.Identity()

        # Metadata branch
        self.metadata_branch = nn.Sequential(
            nn.BatchNorm1d(num_metadata_features),
            nn.Linear(num_metadata_features, config['meta_fc1_neurons']),
            nn.ReLU(),
            nn.Dropout(config['meta_dropout']),
            nn.Linear(config['meta_fc1_neurons'], config['meta_fc2_neurons']),
            nn.ReLU()
        )

        # Combined branch
        combined_input_features = self.resnet_feature_dim + config['meta_fc2_neurons']
        self.combined_head = nn.Sequential(
            nn.Linear(combined_input_features, config['comb_fc1_neurons']),
            nn.ReLU(),
            nn.Linear(config['comb_fc1_neurons'], config['comb_fc2_neurons']),
            nn.ReLU(),
            nn.Dropout(config['comb_dropout']),
            nn.Linear(config['comb_fc2_neurons'], 1)
        )

    def forward(self, image_input: torch.Tensor, metadata_input: torch.Tensor) -> torch.Tensor:
        image_features = self.resnet_backbone(image_input)
        meta_features = self.metadata_branch(metadata_input)
        combined_features = torch.cat((image_features, meta_features), dim=1)
        logits = self.combined_head(combined_features)
        return logits


class um_nn(nn.Module):
    def __init__(self, config):
        super(um_nn, self).__init__()
        num_metadata_features = len(config.get("metadata_cols", []))

        self.network = nn.Sequential(
            nn.BatchNorm1d(num_metadata_features),
            nn.Linear(num_metadata_features, config['meta_fc1_neurons']),
            nn.ReLU(True),
            nn.Dropout(config['meta_dropout']),
            nn.Linear(config['meta_fc1_neurons'], config['meta_fc2_neurons']),
            nn.ReLU(True),
            nn.Linear(config['meta_fc2_neurons'], config['head_neurons']),
            nn.ReLU(True),
            nn.Dropout(config['head_dropout']),
            nn.Linear(config['head_neurons'], 1)
        )

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        return self.network(input_data)
