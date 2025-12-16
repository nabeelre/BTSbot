import torch
import torch.nn as nn

import re
import timm
import json
import os.path as path


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
            nn.Dropout(config['dropout']),
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
            nn.Dropout(config['dropout']),
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


class mm_cnn(nn.Module):
    def __init__(self, config):
        super(mm_cnn, self).__init__()
        num_metadata_features = len(config.get("metadata_cols", []))

        # Image branch (CNN)
        self.conv_layers = nn.Sequential(
            # First convolutional block
            nn.Conv2d(3, config['conv1_channels'],
                      kernel_size=config['conv_kernel'], padding='same'),
            nn.ReLU(),
            nn.Conv2d(config['conv1_channels'], config['conv1_channels'],
                      kernel_size=config['conv_kernel'], padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(config['conv_dropout1']),

            # Second convolutional block
            nn.Conv2d(config['conv1_channels'], config['conv2_channels'],
                      kernel_size=config['conv_kernel'], padding='same'),
            nn.ReLU(),
            nn.Conv2d(config['conv2_channels'], config['conv2_channels'],
                      kernel_size=config['conv_kernel'], padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout2d(config['conv_dropout2']),
            nn.Flatten()
        )
        self.conv_feature_dim = config['conv2_channels'] * (config.get('image_size', 63) // 8) ** 2

        # Metadata branch
        self.metadata_branch = nn.Sequential(
            nn.BatchNorm1d(num_metadata_features),
            nn.Linear(num_metadata_features, config['meta_fc1_neurons']),
            nn.ReLU(),
            nn.Dropout(config['meta_dropout']),
            nn.Linear(config['meta_fc1_neurons'], config['meta_fc2_neurons']),
            nn.ReLU()
        )

        combined_input_features = self.conv_feature_dim + config['meta_fc2_neurons']
        self.combined_head = nn.Sequential(
            nn.Linear(combined_input_features, config['comb_fc1_neurons']),
            nn.ReLU(),
            nn.Linear(config['comb_fc1_neurons'], config['comb_fc2_neurons']),
            nn.ReLU(),
            nn.Dropout(config['comb_dropout']),
            nn.Linear(config['comb_fc2_neurons'], 1)
        )

    def forward(self, image_input: torch.Tensor, metadata_input: torch.Tensor) -> torch.Tensor:
        conv_features = self.conv_layers(image_input)
        meta_features = self.metadata_branch(metadata_input)
        combined_features = torch.cat((conv_features, meta_features), dim=1)
        logits = self.combined_head(combined_features)
        return logits


class um_cnn(nn.Module):
    def __init__(self, config):
        super(um_cnn, self).__init__()

        # CNN backbone
        self.conv_layers = nn.Sequential(
            # First convolutional block
            nn.Conv2d(3, config['conv1_channels'],
                      kernel_size=config['conv_kernel'], padding='same'),
            nn.ReLU(),
            nn.Conv2d(config['conv1_channels'], config['conv1_channels'],
                      kernel_size=config['conv_kernel'], padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(config['conv_dropout1']),

            # Second convolutional block
            nn.Conv2d(config['conv1_channels'], config['conv2_channels'],
                      kernel_size=config['conv_kernel'], padding='same'),
            nn.ReLU(),
            nn.Conv2d(config['conv2_channels'], config['conv2_channels'],
                      kernel_size=config['conv_kernel'], padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout2d(config['conv_dropout2']),
            nn.Flatten()
        )

        conv_feature_dim = config['conv2_channels'] * (config.get('image_size', 63) // 8) ** 2

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(conv_feature_dim, config['fc1_neurons']),
            nn.ReLU(),
            nn.Linear(config['fc1_neurons'], config['fc2_neurons']),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['fc2_neurons'], 1)
        )

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        features = self.conv_layers(input_data)
        return self.head(features)


class um_nn(nn.Module):
    def __init__(self, config):
        super(um_nn, self).__init__()
        num_metadata_features = len(config.get("metadata_cols", []))

        self.network = nn.Sequential(
            nn.BatchNorm1d(num_metadata_features),
            nn.Linear(num_metadata_features, config['meta_fc1_neurons']),
            nn.ReLU(),
            nn.Dropout(config['meta_dropout']),
            nn.Linear(config['meta_fc1_neurons'], config['meta_fc2_neurons']),
            nn.ReLU(),
            nn.Linear(config['meta_fc2_neurons'], 1)
        )

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        return self.network(input_data)


class frozen_fusion(nn.Module):
    @staticmethod
    def remove_branch_head(model, model_name):
        if model_name == "um_nn":
            model.network = nn.Sequential(
                *list(model.network.children())[:-2]
            )
            emb_dim = model.network[-1].out_features
        elif model_name == "MaxViT":
            emb_dim = model.maxvit.head[1].in_features
            model.maxvit.head = nn.Sequential(
                *list(model.maxvit.head.children())[0:1]
            )
        elif model_name == "ConvNeXt":
            model.convnext.head = nn.Sequential(
                *list(model.convnext.head.children())[0:3]
            )
            emb_dim = model.convnext.head[1].normalized_shape[0]
        elif model_name == "um_cnn":
            emb_dim = model.head[0].in_features
            model.head = nn.Identity()
        else:
            raise ValueError(f"Model {model_name} not supported")

        return model, emb_dim

    @staticmethod
    def load_BTSbot_model(model_dir, train_config=None, skip_load_state=False):
        if train_config is None:
            with open(path.join(model_dir, "report.json"), 'r') as f:
                train_config = json.load(f)['train_config']

        try:
            model_type = globals()[train_config['model_name']]
        except KeyError:
            print(f"Could not find model of name {train_config['model_name']}")
            exit(0)
        model = model_type(train_config)
        if not skip_load_state:
            model.load_state_dict(torch.load(path.join(model_dir, "best_model.pth")))
        model, emb_dim = frozen_fusion.remove_branch_head(model, train_config['model_name'])

        return model, emb_dim

    def __init__(self, config):
        super(frozen_fusion, self).__init__()
        # Image branch
        image_model_config = config.get('image_model_config', None)
        self.image_branch, img_emb_dim = frozen_fusion.load_BTSbot_model(
            config['image_model_dir'], train_config=image_model_config,
            skip_load_state=config.get('skip_load_state', False)
        )

        # Metadata branch
        meta_model_config = config.get('meta_model_config', None)
        self.meta_branch, meta_emb_dim = frozen_fusion.load_BTSbot_model(
            config['meta_model_dir'], train_config=meta_model_config,
            skip_load_state=config.get('skip_load_state', False)
        )

        # Combined branch
        combined_dim = img_emb_dim + meta_emb_dim
        self.combined_head = nn.Sequential(
            nn.Linear(combined_dim, config['comb_fc1_neurons']),
            nn.ReLU(),
            nn.Linear(config['comb_fc1_neurons'], config['comb_fc2_neurons']),
            nn.ReLU(),
            nn.Dropout(config['comb_dropout']),
            nn.Linear(config['comb_fc2_neurons'], 1)
        )

    def forward(self, image_input: torch.Tensor, metadata_input: torch.Tensor) -> torch.Tensor:
        img_features = self.image_branch(image_input)
        meta_features = self.meta_branch(metadata_input)
        combined_features = torch.cat((img_features, meta_features), dim=1)
        logits = self.combined_head(combined_features)
        return logits
