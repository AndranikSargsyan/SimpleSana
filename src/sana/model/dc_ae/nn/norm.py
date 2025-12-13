from typing import Optional

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from ..nn.triton_rms_norm import TritonRMSNorm2dFunc
from ..utils import build_kwargs_from_config

__all__ = ["LayerNorm2d", "TritonRMSNorm2d", "build_norm", "set_norm_eps"]


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x - torch.mean(x, dim=1, keepdim=True)
        out = out / torch.sqrt(torch.square(out).mean(dim=1, keepdim=True) + self.eps)
        if self.elementwise_affine:
            out = out * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return out


class TritonRMSNorm2d(nn.LayerNorm):
    def zero_out(self):
        nn.init.constant_(self.weight, 0)
        nn.init.constant_(self.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_numel = x.numel()
        if input_numel >= 1 << 31:
            num_chunks = (input_numel - 1) // (1 << 31) + 1
            output = []
            for x_chunk in x.chunk(num_chunks, dim=2):
                output.append(TritonRMSNorm2dFunc.apply(x_chunk.contiguous(), self.weight, self.bias, self.eps))
            output = torch.cat(output, dim=2)
            return output
        else:
            return TritonRMSNorm2dFunc.apply(x.contiguous(), self.weight, self.bias, self.eps)


class RMSNorm2d(nn.Module):
    def __init__(
        self, num_features: int, eps: float = 1e-5, elementwise_affine: bool = True, bias: bool = True
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = torch.nn.parameter.Parameter(torch.empty(self.num_features))
            if bias:
                self.bias = torch.nn.parameter.Parameter(torch.empty(self.num_features))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x / torch.sqrt(torch.square(x.float()).mean(dim=1, keepdim=True) + self.eps)).to(x.dtype)
        if self.elementwise_affine:
            x = x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return x


# register normalization function here
REGISTERED_NORM_DICT: dict[str, type] = {
    "bn2d": nn.BatchNorm2d,
    "ln": nn.LayerNorm,
    "ln2d": LayerNorm2d,
    "trms2d": TritonRMSNorm2d,
    "rms2d": RMSNorm2d,
}


def build_norm(name="bn2d", num_features=None, **kwargs) -> Optional[nn.Module]:
    if name in ["ln", "ln2d", "trms2d"]:
        kwargs["normalized_shape"] = num_features
    else:
        kwargs["num_features"] = num_features
    if name in REGISTERED_NORM_DICT:
        norm_cls = REGISTERED_NORM_DICT[name]
        args = build_kwargs_from_config(kwargs, norm_cls)
        return norm_cls(**args)
    else:
        return None


def set_norm_eps(model: nn.Module, eps: Optional[float] = None) -> None:
    for m in model.modules():
        if isinstance(m, (nn.GroupNorm, nn.LayerNorm, _BatchNorm)):
            if eps is not None:
                m.eps = eps
