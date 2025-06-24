import torch.nn as nn
from typing import Type

# A mapping from string identifiers to their corresponding nn.Module classes.
# This makes the factory function cleaner and more extensible.
_NONLINEARITIES: dict[str, Type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
}


def make_nlrity(nonlinearity: str = 'relu') -> nn.Module:
    """
    Creates an instance of a nonlinearity module.

    Args:
        nonlinearity (str): The name of the nonlinearity.
                            Supported values are 'relu' and 'gelu'.
                            Defaults to 'relu'.

    Returns:
        nn.Module: An instance of the requested nonlinearity module.

    Raises:
        ValueError: If the `nonlinearity` name is not supported.
    """
    nl_key = nonlinearity.lower()
    if nl_key not in _NONLINEARITIES:
        raise ValueError(
            f"Unknown nonlinearity '{nonlinearity}'. "
            f"Supported options are: {list(_NONLINEARITIES.keys())}"
        )

    # Instantiate and return the module
    return _NONLINEARITIES[nl_key]()