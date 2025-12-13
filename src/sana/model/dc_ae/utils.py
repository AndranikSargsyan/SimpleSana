from inspect import signature
from typing import Any, Callable, Optional, Union

import torch
import torch.nn.functional as F

__all__ = [
    "get_same_padding",
    "resize",
    "build_kwargs_from_config",
]


def get_same_padding(kernel_size: Union[int, tuple[int, ...]]) -> Union[int, tuple[int, ...]]:
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, "kernel size should be odd number"
        return kernel_size // 2


def resize(
    x: torch.Tensor,
    size: Optional[Any] = None,
    scale_factor: Optional[list[float]] = None,
    mode: str = "bicubic",
    align_corners: Optional[bool] = False,
) -> torch.Tensor:
    if mode in {"bilinear", "bicubic"}:
        return F.interpolate(
            x,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
        )
    elif mode in {"nearest", "area"}:
        return F.interpolate(x, size=size, scale_factor=scale_factor, mode=mode)
    else:
        raise NotImplementedError(f"resize(mode={mode}) not implemented.")


def build_kwargs_from_config(config: dict, target_func: Callable) -> dict[str, Any]:
    valid_keys = list(signature(target_func).parameters)
    kwargs = {}
    for key in config:
        if key in valid_keys:
            kwargs[key] = config[key]
    return kwargs

