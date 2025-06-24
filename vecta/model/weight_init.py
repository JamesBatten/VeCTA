import torch
import torch.nn as nn
from typing import Callable


def make_weights_init(
    nonlinearity: str = 'relu',
    initialisation: str = 'xavier'
) -> Callable[[nn.Module], None]:
    """
    Creates a function to initialize module weights and biases.

    This is a higher-order function that returns a closure. The returned
    function is designed to be used with `module.apply()`.

    The initialization strategy is as follows:
    - For nn.Linear and nn.Conv2d layers:
        - Weights are initialized using Xavier Uniform.
        - Biases are initialized to a small constant (0.1) if the subsequent
          nonlinearity is 'relu', to help prevent dead neurons. Otherwise,
          they are initialized from a uniform distribution U(-1, 1).
    - For nn.GroupNorm and nn.LayerNorm layers:
        - Weights are initialized to 1.
        - Biases are initialized from a uniform distribution U(-1, 1).

    Args:
        nonlinearity (str): The name of the nonlinearity that follows the
                            layer. This affects bias initialization.
                            Defaults to 'relu'.
        initialisation (str): The name of the weight initialization method.
                              Currently, only 'xavier' is supported.
                              Defaults to 'xavier'.

    Returns:
        Callable[[nn.Module], None]: A function that takes a module `m` and
                                     initializes its parameters in-place.

    Raises:
        ValueError: If an unsupported `initialisation` method is requested.
    """
    if initialisation.lower() != 'xavier':
        raise ValueError(
            f"Unknown initialisation method '{initialisation}'. "
            "Only 'xavier' is currently supported."
        )

    def weights_init(m: nn.Module) -> None:
        """Applies initialization to a single module."""
        # Handle Linear and Convolutional layers
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                if nonlinearity.lower() == 'relu':
                    # Small constant bias for ReLU to avoid dead neurons
                    torch.nn.init.constant_(m.bias, 0.1)
                else:
                    # Note: U(-1, 1) is a wide range for biases.
                    # A more common practice is zero initialization.
                    torch.nn.init.uniform_(m.bias, -1.0, 1.0)

        # Handle Normalization layers
        elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
            if getattr(m, 'weight', None) is not None:
                nn.init.constant_(m.weight, 1.0)
            if getattr(m, 'bias', None) is not None:
                # Note: U(-1, 1) is a wide range for biases.
                # A more common practice is zero initialization.
                torch.nn.init.uniform_(m.bias, -1.0, 1.0)

    return weights_init