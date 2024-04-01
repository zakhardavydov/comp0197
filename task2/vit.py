from typing import Any

import torch.nn as nn

from torchvision.models import VisionTransformer


def get_vit(params: dict[str, Any]) -> nn.Module:
    """
    Get ViT with given params
    """
    model = VisionTransformer(**params)
    return model
