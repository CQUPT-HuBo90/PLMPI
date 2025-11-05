import timm

import torch.nn as nn

from pathlib import Path
from .utils import activations, forward_default, get_activation

# from ..external.next_vit.classification.nextvit import *


def forward_next_vit(pretrained, x):
    return forward_default(pretrained, x, "forward")


def _make_next_vit_backbone(
        model,
        hooks=[2, 6, 36, 39],
):
    pretrained = nn.Module()

    pretrained.model = model
    pretrained.model.stages[hooks[0]].register_forward_hook(get_activation("1"))
    pretrained.model.stages[hooks[1]].register_forward_hook(get_activation("2"))
    pretrained.model.stages[hooks[2]].register_forward_hook(get_activation("3"))
    pretrained.model.stages[hooks[3]].register_forward_hook(get_activation("4"))

    pretrained.activations = activations

    return pretrained

def _make_pretrained_next_vit_large_6m(hooks=None):
    model = timm.create_model("nextvit_large")    
    # hooks直接使用stages的索引（0-3），每个值对应一个stage
    hooks = [0, 1, 2, 3]  # 正确取值：分别对应stage[0]到stage[3]
    return _make_next_vit_backbone(
        model,
        hooks=hooks,
    )
