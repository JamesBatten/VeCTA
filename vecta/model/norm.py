import torch.nn as nn
from typing import Optional, Type


def make_norm(
    out_dims: int,
    norm: str = 'groupnorm',
    numgroups: int = 8
) -> nn.Module:
    """
    Creates an instance of a normalization layer.

    Args:
        out_dims (int): The number of channels or features of the input.
        norm (str): The name of the normalization layer.
                    Supported values: 'groupnorm', 'layernorm'.
        numgroups (int): The number of groups for GroupNorm. This argument
                         is ignored for other normalization types.

    Returns:
        nn.Module: An instantiated normalization layer.

    Raises:
        ValueError: If the `norm` type is not supported or if `out_dims` is
                    not divisible by `numgroups` for GroupNorm.
    """
    norm_key = norm.lower()
    if norm_key == 'groupnorm':
        if out_dims % numgroups != 0:
            raise ValueError(
                f"For GroupNorm, out_dims ({out_dims}) must be divisible by "
                f"numgroups ({numgroups})."
            )
        return nn.GroupNorm(numgroups, out_dims)
    elif norm_key == 'layernorm':
        return nn.LayerNorm(out_dims)

    raise ValueError(
        f"Unknown normalization type '{norm}'. "
        "Supported options are: 'groupnorm', 'layernorm'."
    )


def make_norm_layer(norm: Optional[str] = 'groupnorm') -> Optional[Type[nn.Module]]:
    """
    Returns the class of a normalization layer.

    Note: This function appears to be unused in the provided project context.
          If it is used, its functionality is preserved.

    Args:
        norm (Optional[str]): The name of the normalization layer.
                              If None, returns None.
                              Supported values: 'groupnorm', 'layernorm'.

    Returns:
        Optional[Type[nn.Module]]: The normalization layer class, or None.

    Raises:
        ValueError: If the `norm` type is not None and is not supported.
    """
    if norm is None:
        return None

    norm_map: dict[str, Type[nn.Module]] = {
        'groupnorm': nn.GroupNorm,
        'layernorm': nn.LayerNorm,
    }

    norm_key = norm.lower()
    if norm_key not in norm_map:
        raise ValueError(
            f"Unknown normalization type '{norm}'. "
            f"Supported options are: {list(norm_map.keys())}."
        )

    return norm_map[norm_key]