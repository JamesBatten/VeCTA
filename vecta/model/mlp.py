from __future__ import annotations

import torch
import torch.nn as nn

from vecta.model.nonlinearity import make_nlrity
from vecta.model.norm import make_norm
from vecta.model.weight_init import make_weights_init


class MLP2(nn.Module):
    """
    A two-layer Multi-Layer Perceptron (MLP).

    This module consists of an input linear layer, a nonlinearity, an optional
    normalization layer, and an output linear layer. It uses nn.Sequential for
    a clean and idiomatic implementation.

    The architecture is:
    Linear(in_dims, hidden_dims) -> Non-linearity -> [Normalization] -> Linear(hidden_dims, out_dims)
    """

    def __init__(
        self,
        in_dims: int,
        hidden_dims: int,
        out_dims: int,
        nonlinearity: str = 'relu',
        bias_1: bool = True,
        bias_2: bool = True,
        norm_1: str | None = None,
        numgroups: int = 8
    ) -> None:
        """
        Initializes the MLP2 module.

        Args:
            in_dims (int): Number of input features.
            hidden_dims (int): Number of units in the hidden layer.
            out_dims (int): Number of output features.
            nonlinearity (str): The name of the nonlinearity to use after the
                first linear layer (e.g., 'relu', 'gelu'). Defaults to 'relu'.
            bias_1 (bool): Whether to include a bias term in the first linear
                layer. Defaults to True.
            bias_2 (bool): Whether to include a bias term in the second linear
                layer. Defaults to True.
            norm_1 (str | None): The type of normalization to apply after the
                nonlinearity. Supported values are 'groupnorm', 'layernorm', or
                None to disable. Defaults to None.
            numgroups (int): The number of groups for GroupNorm. This is ignored
                if `norm_1` is not 'groupnorm'. Defaults to 8.
        """
        super().__init__()

        # Build the network layers sequentially.
        layers = [
            nn.Linear(in_dims, hidden_dims, bias=bias_1),
            make_nlrity(nonlinearity)
        ]

        # Conditionally add the normalization layer.
        if norm_1 is not None:
            # Note: The original implementation hardcoded 'groupnorm'. This
            # version correctly uses the `norm_1` parameter.
            layers.append(make_norm(
                hidden_dims, norm=norm_1, numgroups=numgroups
            ))

        layers.append(nn.Linear(hidden_dims, out_dims, bias=bias_2))

        # Use nn.Sequential to chain the layers together.
        self.network = nn.Sequential(*layers)

        # Apply custom weight initialization to all submodules.
        self.network.apply(make_weights_init(nonlinearity))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the MLP.

        Args:
            x (torch.Tensor): The input tensor of shape (*, in_dims).

        Returns:
            torch.Tensor: The output tensor of shape (*, out_dims).
        """
        return self.network(x)