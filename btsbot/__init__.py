"""
BTSbot: Multi-modal convolutional neural network for automating supernova identification
and follow-up in the Zwicky Transient Facility (ZTF) Bright Transient Survey (BTS).
"""

__version__ = "2.0.0"

# Core modules
from . import architectures
from . import utils
from . import alert_utils
from . import from_HF

# Main classes and functions
from .utils import FlexibleDataset, RandomRightAngleRotation, make_report
from .architectures import (
    MaxViT,
    ConvNeXt,
    mm_MaxViT,
    mm_ConvNeXt,
    mm_cnn,
    um_cnn,
    um_nn,
    frozen_fusion,
)
from .from_HF import download_HF_model, load_HF_model

__all__ = [
    "__version__",
    "architectures",
    "utils",
    "alert_utils",
    "FlexibleDataset",
    "RandomRightAngleRotation",
    "make_report",
    "SwinV2",
    "MaxViT",
    "ConvNeXt",
    "mm_SwinV2",
    "mm_MaxViT",
    "mm_ConvNeXt",
    "mm_ResNet",
    "mm_cnn",
    "um_cnn",
    "um_nn",
    "frozen_fusion",
    "get_model_image_size",
    "download_HF_model",
    "load_HF_model",
]
