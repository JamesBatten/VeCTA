"""
Utility functions for data type conversion and feature engineering.

This module provides helper functions for:
- Converting data structures (including lists, dicts, NumPy arrays) to/from PyTorch tensors.
- Performing positional encoding ("lifting") on features using sinusoidal functions.
"""
from __future__ import annotations

import typing

import numpy as np
import numpy.typing as npt
import torch

if typing.TYPE_CHECKING:
    # Define a recursive type for nested structures used in conversion functions.
    RecursiveStructure = typing.Union[
        torch.Tensor,
        npt.NDArray,
        int,
        float,
        typing.List['RecursiveStructure'],
        typing.Dict[str, 'RecursiveStructure'],
    ]


def to_array(
    x: RecursiveStructure
) -> npt.NDArray | list | dict:
    """
    Recursively converts a PyTorch Tensor or nested structure to a NumPy array.

    Args:
        x: The input data, which can be a PyTorch Tensor, NumPy array, list,
           dict, or scalar.

    Returns:
        The converted data in NumPy format. Dictionaries and lists retain their
        structure but with tensors converted to arrays.

    Raises:
        TypeError: If an unsupported data type is encountered.
    """
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, list):
        return [to_array(item) for item in x]
    if isinstance(x, dict):
        return {key: to_array(value) for key, value in x.items()}
    if isinstance(x, (int, float)):
        return np.array(x)

    raise TypeError(f"Unsupported type for conversion to NumPy array: {type(x)}")


def take_along_axis(x: torch.Tensor | npt.NDArray, index: int, axis: int) -> torch.Tensor | npt.NDArray:
    """
    Selects a single slice from a tensor/array along a specified axis.

    This is a simplified, more efficient alternative to `torch.index_select` or
    `np.take_along_axis` for selecting a single index.

    Args:
        x: The source tensor or array.
        index: The integer index of the slice to select.
        axis: The axis from which to select the slice.

    Returns:
        A new tensor/array containing the selected slice, with the specified
        axis removed (squeezed).
    """
    slicing = [slice(None)] * x.ndim
    slicing[axis] = index
    return x[tuple(slicing)]


def liftsin(x: torch.Tensor | npt.NDArray, oct: int) -> torch.Tensor | npt.NDArray:
    """Computes the sine positional encoding for a given octave."""
    if isinstance(x, np.ndarray):
        return np.sin(2 * np.pi * oct * x).astype(np.float32)
    if isinstance(x, torch.Tensor):
        return torch.sin(2 * np.pi * oct * x).float()
    raise TypeError(f"Unsupported type for liftsin: {type(x)}")


def liftcos(x: torch.Tensor | npt.NDArray, oct: int) -> torch.Tensor | npt.NDArray:
    """Computes the cosine positional encoding for a given octave."""
    if isinstance(x, np.ndarray):
        return np.cos(2 * np.pi * oct * x).astype(np.float32)
    if isinstance(x, torch.Tensor):
        return torch.cos(2 * np.pi * oct * x).float()
    raise TypeError(f"Unsupported type for liftcos: {type(x)}")


def lift(x: torch.Tensor | npt.NDArray, oct: int, lifter: str) -> torch.Tensor | npt.NDArray:
    """
    Applies a specified lifting function (positional encoding).

    Args:
        x: The input tensor or array.
        oct: The octave (frequency multiplier).
        lifter: The lifting function to apply, either 'sin' or 'cos'.

    Returns:
        The positionally encoded tensor or array.

    Raises:
        ValueError: If an unsupported lifter is specified.
    """
    lifter_lower = lifter.lower()
    if lifter_lower == "sin":
        return liftsin(x, oct)
    if lifter_lower == "cos":
        return liftcos(x, oct)
    raise ValueError(f"Unknown lifter '{lifter}'. Supported lifters are 'sin', 'cos'.")


def add_octaves(
    x: torch.Tensor | npt.NDArray[np.float32],
    octaves: list[int],
    dim: int,
    channels: list[int] | None = None,
    lifters: list[str] = ['sin', 'cos'],
    include_base: bool = True
) -> torch.Tensor | npt.NDArray[np.float32]:
    """
    Applies positional encoding to specified channels of a tensor or array.

    This function expands the feature representation of an input by concatenating
    sinusoidal encodings (octaves) of its values.

    Example:
        Input: x tensor of shape (b, 3, h, w), octaves=[1, 2, 4], dim=1,
               channels=[1, 2] (for Y and Z axes).
        Output: Tensor of shape (b, 3 + 2 * len(octaves) * len(lifters), h, w).
                The original 3 channels + encoded channels for Y and Z.

    Args:
        x: The input tensor or array.
        octaves: A list of integer frequencies for the sinusoidal functions.
        dim: The dimension along which the channels are stacked.
        channels: A list of indices for the channels within `dim` to encode.
            If None, all channels are encoded. Defaults to None.
        lifters: A list of lifting functions to apply, e.g., ['sin', 'cos'].
            Defaults to ['sin', 'cos'].
        include_base: If True, the original input `x` is concatenated with the
            newly generated octave features. If False, only the new features
            are returned. Defaults to True.

    Returns:
        The tensor or array with added octave features.

    Raises:
        ValueError: If the `dim` is out of bounds for the input `x`.
    """
    if not (0 <= dim < x.ndim):
        raise ValueError(f"Dimension {dim} is out of bounds for input with {x.ndim} dimensions.")

    if channels is None:
        channels = list(range(x.shape[dim]))

    new_features = []
    for c_idx in channels:
        if not (0 <= c_idx < x.shape[dim]):
            raise ValueError(f"Channel index {c_idx} is out of range for dimension {dim} with size {x.shape[dim]}.")

        channel_slice = take_along_axis(x, c_idx, dim)
        for oct in octaves:
            for lft in lifters:
                new_features.append(lift(channel_slice, oct, lft))

    if not new_features:
        return x  # Return original if no new features were generated

    if isinstance(x, np.ndarray):
        stacked_features = np.stack(new_features, axis=dim)
        return np.concatenate((x, stacked_features), axis=dim) if include_base else stacked_features
    elif isinstance(x, torch.Tensor):
        stacked_features = torch.stack(new_features, dim=dim)
        return torch.cat((x, stacked_features), dim=dim) if include_base else stacked_features
    else:
        # This case should not be reached if input is one of the hinted types.
        raise TypeError(f"Unsupported type for add_octaves: {type(x)}")


def to_tensor(
    x: RecursiveStructure | None,
    device: torch.device | str | None = None
) -> torch.Tensor | list | dict | None:
    """
    Recursively converts a NumPy array or nested structure to a PyTorch Tensor.

    Args:
        x: The input data, which can be a NumPy array, PyTorch Tensor, list,
           dict, scalar, or None.
        device: The target device for the tensor (e.g., 'cpu', 'cuda:0').
                If None, tensors are created on the default device.

    Returns:
        The converted data as a PyTorch Tensor. Dictionaries and lists retain
        their structure but with arrays converted to tensors. Returns None if
        the input is None.

    Raises:
        TypeError: If an unsupported data type is encountered.
    """
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.to(device) if device else x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device)
    if isinstance(x, list):
        return [to_tensor(item, device=device) for item in x]
    if isinstance(x, dict):
        return {key: to_tensor(value, device=device) for key, value in x.items()}
    if isinstance(x, (int, float, np.number)):
        return torch.tensor(x, device=device)

    raise TypeError(f"Unsupported type for conversion to torch.Tensor: {type(x)}")