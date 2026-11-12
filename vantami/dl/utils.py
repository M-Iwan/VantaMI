from typing import Literal

from torch import nn


ActivationFunctions = Literal["relu", "leaky_relu", "gelu", "tanh", "silu"]


def get_activation_fn(name: ActivationFunctions = "silu") -> nn.Module:
    """
    Return a built-in activation module by name (no learnable parameters).

    Parameters
    ----------
    name: ActivationFunctions
        One of relu, leaky_relu, gelu, tanh, silu. Matching is case-insensitive.

    Returns
    -------
    act: nn.Module
        A fresh activation instance (e.g. nn.ReLU()).
    """
    act_fn = {
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(),
        "gelu": nn.GELU(),
        "tanh": nn.Tanh(),
        "silu": nn.SiLU(),
    }
    if (name := name.lower().strip()) not in act_fn.keys():
        raise ValueError(f"Unsupported activation function: {name}")
    return act_fn[name]
