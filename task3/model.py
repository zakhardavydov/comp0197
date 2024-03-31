from typing import Any

import torch.nn as nn

from torchvision.models import VisionTransformer
from original import OriginalNet


def get_model(params: dict[str, Any], original: bool = False) -> nn.Module:
    """
    Model loader that can either take original model architecture or ViT
    """
    if original:
        return OriginalNet()
    return VisionTransformer(**params)
