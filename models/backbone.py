# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding
from .c3d import C3D
from .resnet import resnet_tdcnn
from .i3dpt import build_base_i3d


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int):
        super().__init__()

        self.body = backbone
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        m = tensor_list.mask
        assert m is not None
        mask = F.interpolate(m[None].float(), size=xs.shape[-3:]).to(torch.bool)[0]
        out = NestedTensor(xs, mask)

        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool):
        if name == "c3d":
            backbone = C3D()
            backbone = nn.Sequential(*list(backbone.features._modules.values())[:-1])
            num_channels = 512
            model_path = "./pretrained_model/c3d_sports1M.pth"
            state_dict = torch.load(model_path)
            try:
                backbone.load_state_dict({k: v for k, v in state_dict.items() if (k in backbone.state_dict())})
            except Exception as e:
                print(e)
        elif name == "resnet18":
            num_channels = 256
            backbone = resnet_tdcnn(depth=18)
        elif name == "resnet34":
            num_channels = 256
            backbone = resnet_tdcnn(depth=34)
        elif name == "I3D":
            num_channels = 832
            backbone = build_base_i3d("./pretrained_model/i3d_kinetics.pth")

        super().__init__(backbone, train_backbone, num_channels)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor, supports, shot):
        xs = self[0](tensor_list)
        if shot > 0:
            B, S, C, T, H, W = supports.shape
            features_support = self[0].body(supports.view(-1, C, T, H, W))
            _, C, T, H, W = features_support.shape
            _features_support = features_support.view(B, S, C, T, H, W)
        else:
            _features_support = None
        out: List[NestedTensor] = []
        pos = []
        out.append(xs)
        # position encoding
        pos.append(self[1](xs).to(xs.tensors.dtype))

        return out, pos, _features_support


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    backbone = Backbone(args.net, train_backbone)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model

