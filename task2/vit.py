from typing import Any

import torch.nn as nn

from torchvision.models import VisionTransformer


def get_vit(params: dict[str, Any]) -> nn.Module:
    model = VisionTransformer(**params)
    return model
