from __future__ import annotations

from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

from vecta.common.utils import add_octaves, to_array
from vecta.model.mlp import MLP


class VesselDecoder(nn.Module):
    """
    Decodes a latent vector into a sequence of vessel points.

    This module reconstructs a vessel segment from a latent vector `v` at
    specified time coordinates `t_b`. It supports several operational modes
    controlled by its configuration.

    The decoding process can operate in two main modes:
    1.  'default': Directly predicts the point coordinates (x, y, z, r).
    2.  'deviation': Predicts a deviation from a linear interpolation between
        two endpoints (`pa` and `pb`), allowing for more detailed and stable
        reconstructions.

    It can also explicitly decode the vessel's start and end points from the
    latent vector, which are then used in 'deviation' mode during inference.

    Configuration options (passed as a dictionary):
        in_dims (int): Input dimensions for the main MLP.
        embed_dims (int): Dimension of the input latent vector.
        hidden_dims (int): Hidden layer dimensions for all MLPs.
        out_dims (int): Output dimensions (typically 4 for x,y,z,r).
        n_layers (int): Number of layers for the MLPs.
        grid_size (int): Resolution for inverting positional encoding.
        activation (str): Activation function for MLPs (e.g., 'gelu').
        octaves (list[int]): Frequencies for positional encoding.
        wrap_domain (tuple[float, float]): Domain for positional encoding inversion.
        mode (Literal['default', 'deviation']): Main operating mode.
        modulation (str): Modulation function for 'deviation' mode.
        decode_endpoints (bool): If True, MLPs to decode endpoints are created.
        rad_mode (Literal['lift', 'log']): How radius is encoded/decoded.
        head_mode (Literal['default', 'separate', 'separate_relu']):
            Defines the MLP architecture for processing the latent vector.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__()
        if config is None:
            config = VesselDecoder.default_config()
        self.config = config

        self._validate_config()
        self._build_networks()

        if self.config['decode_endpoints']:
            self._build_pos_array()
            # Pre-calculate dimensions to avoid magic numbers later
            self._lifted_endpoint_dim, self._pos_lifted_dim, self._rad_lifted_dim = \
                self._calculate_endpoint_dims()

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        """Returns the default configuration dictionary for the decoder."""
        return {
            'in_dims': 76,  # 64 (bottleneck) + 12 (time octaves)
            'embed_dims': 64,
            'hidden_dims': 2048,
            'out_dims': 4,  # (x, y, z, r) or (x, y, z, log(1 + r))
            'n_layers': 2,
            'grid_size': 1000,
            'activation': 'gelu',
            'octaves': [1, 2, 4, 8, 16, 32],
            'wrap_domain': (-0.25, 0.25),
            'mode': 'default',  # 'default' or 'deviation'
            'modulation': 'default',  # 'default', 'none', 'sharp1', etc.
            'decode_endpoints': False,
            'rad_mode': 'lift',  # 'lift' or 'log'
            'head_mode': 'default'  # 'default', 'separate', 'separate_relu'
        }

    def _validate_config(self):
        """Validates the configuration dictionary."""
        if self.config['mode'] not in ['default', 'deviation']:
            raise ValueError(f"Invalid mode: {self.config['mode']}")
        if self.config['rad_mode'] not in ['lift', 'log']:
            raise ValueError(f"Invalid rad_mode: {self.config['rad_mode']}")
        if self.config['head_mode'] not in ['default', 'separate', 'separate_relu']:
            raise ValueError(f"Invalid head_mode: {self.config['head_mode']}")
        if self.config['head_mode'] in ['separate', 'separate_relu'] and not self.config['decode_endpoints']:
            raise ValueError("Separate head modes require 'decode_endpoints' to be True.")

    def _build_networks(self):
        """Creates the MLP networks based on the configuration."""
        head_mode = self.config['head_mode']
        # Common MLP arguments
        mlp_args = {
            'hidden_dims': self.config['hidden_dims'],
            'n_layers': self.config['n_layers'],
            'activation': self.config['activation']
        }

        # --- Main Head(s) for decoding the vessel sequence ---
        if head_mode == 'default':
            self.mlp = MLP(self.config['in_dims'], out_dims=self.config['out_dims'], **mlp_args)
        elif head_mode == 'separate':
            in_dims_split = self.config['embed_dims'] // 3 + 12
            self.mlp_pos = MLP(in_dims_split, out_dims=3, **mlp_args)
            self.mlp_rad = MLP(in_dims_split, out_dims=1, **mlp_args)
        elif head_mode == 'separate_relu':
            relu_mlp_args = {
                'hidden_dims': self.config['hidden_dims'], 'out_dims': self.config['embed_dims'],
                'n_layers': 2, 'activation': 'relu', 'norm_1': 'groupnorm', 'numgroups': 8
            }
            self.mlp_pos_relu = MLP(self.config['embed_dims'], **relu_mlp_args)
            self.mlp_pos = MLP(self.config['in_dims'], out_dims=3, **mlp_args)
            self.mlp_rad_relu = MLP(self.config['embed_dims'], **relu_mlp_args)
            self.mlp_rad = MLP(self.config['in_dims'], out_dims=1, **mlp_args)

        # --- Endpoint Head(s) for predicting start/end points ---
        if self.config['decode_endpoints']:
            _, _, out_endpoints_dim = self._calculate_endpoint_dims()
            endpoint_mlp_in_dims = self.config['embed_dims']
            if head_mode == 'separate':
                endpoint_mlp_in_dims = self.config['embed_dims'] // 3

            if head_mode in ['default', 'separate']:
                self.mlp_endpoints = MLP(endpoint_mlp_in_dims, out_dims=2 * out_endpoints_dim, **mlp_args)
            elif head_mode == 'separate_relu':
                relu_mlp_args = {
                    'hidden_dims': self.config['hidden_dims'], 'out_dims': self.config['embed_dims'],
                    'n_layers': 2, 'activation': 'relu', 'norm_1': 'groupnorm', 'numgroups': 8
                }
                self.mlp_endpoints_relu = MLP(self.config['embed_dims'], **relu_mlp_args)
                self.mlp_endpoints = MLP(self.config['embed_dims'], out_dims=2 * out_endpoints_dim, **mlp_args)

    def _build_pos_array(self):
        """Pre-computes the positional array for encoding inversion."""
        self.pos_array = VesselDecoder.make_pos_array_domain(
            self.config['grid_size'], self.config['wrap_domain'], self.config['octaves']
        )
        self.pos_array = self.pos_array.astype(np.float32)

    def _calculate_endpoint_dims(self) -> tuple[int, int, int]:
        """Calculates the feature dimensions for lifted endpoints."""
        octaves_len = len(self.config['octaves'])
        pos_lifted_dim = 6 * octaves_len  # 3 dims (x,y,z) * 2 lifters (sin,cos)
        if self.config['rad_mode'] == 'lift':
            rad_lifted_dim = 2 * octaves_len  # 1 dim (r) * 2 lifters
            total_dim = pos_lifted_dim + rad_lifted_dim
        else:  # 'log' mode
            rad_lifted_dim = 1
            total_dim = pos_lifted_dim + rad_lifted_dim
        return total_dim, pos_lifted_dim, rad_lifted_dim

    # --- Core Forward Logic ---

    def _generate_deviation_output(self, latent_vector: torch.Tensor, time_octaves: torch.Tensor) -> torch.Tensor:
        """
        Generates the raw output from the decoder MLPs before deviation logic.

        This function encapsulates the 'head_mode' logic.
        """
        b, k, _ = time_octaves.shape
        head_mode = self.config['head_mode']

        if head_mode == 'default':
            v_expanded = latent_vector.unsqueeze(1).expand(-1, k, -1)
            mlp_input = torch.cat([v_expanded, time_octaves], dim=-1)
            return self.mlp(mlp_input)

        if head_mode == 'separate':
            v_expanded = latent_vector.unsqueeze(1).expand(-1, k, -1)
            ed = self.config['embed_dims']
            v_pos, v_rad = v_expanded.split([ed // 3, ed // 3], dim=-1, )[:2]

            x_pos = self.mlp_pos(torch.cat([v_pos, time_octaves], dim=-1))
            x_rad = self.mlp_rad(torch.cat([v_rad, time_octaves], dim=-1))
            return torch.cat([x_pos, x_rad], dim=-1)

        if head_mode == 'separate_relu':
            v_pos_relu = self.mlp_pos_relu(latent_vector).unsqueeze(1).expand(-1, k, -1)
            v_rad_relu = self.mlp_rad_relu(latent_vector).unsqueeze(1).expand(-1, k, -1)

            x_pos = self.mlp_pos(torch.cat([v_pos_relu, time_octaves], dim=-1))
            x_rad = self.mlp_rad(torch.cat([v_rad_relu, time_octaves], dim=-1))
            return torch.cat([x_pos, x_rad], dim=-1)
        
        # This line should not be reachable due to _validate_config
        raise RuntimeError(f"Internal error: unhandled head_mode '{head_mode}'")


    def _apply_deviation_modulation(
        self,
        deviation: torch.Tensor,
        time_coords: torch.Tensor,
        start_point: torch.Tensor,
        end_point: torch.Tensor
    ) -> torch.Tensor:
        """Applies the deviation and modulation to the base interpolation."""
        k = time_coords.shape[1]
        t = time_coords.unsqueeze(-1)  # (b, k, 1)
        pa = start_point.unsqueeze(1).expand(-1, k, -1)  # (b, k, 4)
        pb = end_point.unsqueeze(1).expand(-1, k, -1)  # (b, k, 4)

        base_interpolation = pa + (pb - pa) * t
        modulation = self.config['modulation']

        if modulation == 'default':
            mod = t * (1.0 - t)
            return base_interpolation + deviation * mod
        if modulation == 'sharp1':
            # Normalize to peak at 1.0 when t=0.5
            mod = (t**0.1 * (1.0 - t)**0.1) / (0.5**0.2)
            return base_interpolation + deviation * mod
        if modulation == 'none':
            return base_interpolation + deviation
        # Legacy modes for compatibility
        if modulation == 'none_old':
            return base_interpolation + deviation * t
        if modulation == 'sharp1_old':
            mod = (t**0.1 * (1.0 - t)**0.1) / (0.5**0.2)
            return base_interpolation + deviation * t * mod

        raise ValueError(f"Unknown modulation type: {modulation}")

    # --- Public Forward Methods ---

    def forward(
        self,
        v: torch.Tensor,
        t_b: torch.Tensor,
        pa: torch.Tensor | None = None,
        pb: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs the main forward pass, typically used during training.

        Args:
            v: The latent vector batch, shape (b, embed_dims).
            t_b: The time coordinates batch, shape (b, k).
            pa: The start point batch (x,y,z,r), shape (b, 4). Required if
                `mode` is 'deviation'.
            pb: The end point batch (x,y,z,r), shape (b, 4). Required if
                `mode` is 'deviation'.

        Returns:
            If `decode_endpoints` is False, returns the reconstructed points
            tensor of shape (b, k, 4).
            If `decode_endpoints` is True, returns a tuple containing:
            - Reconstructed points tensor (b, k, 4).
            - Predicted lifted start point features (b, lifted_dim).
            - Predicted lifted end point features (b, lifted_dim).
        """
        if self.config['mode'] == 'deviation' and (pa is None or pb is None):
            raise ValueError("'pa' and 'pb' must be provided for 'deviation' mode.")

        # 1. Positional encoding for time coordinates
        time_octaves = add_octaves(
            t_b.unsqueeze(-1), self.config['octaves'], dim=-1, include_base=False
        )

        # 2. Generate raw output from MLPs based on head_mode
        deviation_output = self._generate_deviation_output(v, time_octaves)

        # 3. Apply deviation logic if configured
        if self.config['mode'] == 'deviation':
            x = self._apply_deviation_modulation(deviation_output, t_b, pa, pb)
        else:
            x = deviation_output

        # 4. Handle endpoint decoding if configured
        if self.config['decode_endpoints']:
            lifted_pa, lifted_pb = self._predict_endpoints_lifted(v)
            return x, lifted_pa, lifted_pb

        return x

    # --- Inference-specific Methods ---

    def forward_no_endpoints(self, v: torch.Tensor, t_b: torch.Tensor, return_endpoints: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass for inference.

        This method first decodes the endpoints `pa` and `pb` from the latent
        vector `v`, and then uses them to reconstruct the vessel path. It is
        required that `decode_endpoints` and `mode='deviation'` are True in the
        config.

        Args:
            v: The latent vector batch, shape (b, embed_dims).
            t_b: The time coordinates batch, shape (b, k).
            return_endpoints: If True, returns the decoded `pa` and `pb` along
                              with the reconstructed points.

        Returns:
            If `return_endpoints` is False, returns the reconstructed points
            tensor of shape (b, k, 4).
            If `return_endpoints` is True, returns a tuple containing:
            - Reconstructed points tensor (b, k, 4).
            - Decoded start point tensor `pa` (b, 4).
            - Decoded end point tensor `pb` (b, 4).
        """
        if not self.config['decode_endpoints'] or self.config['mode'] != 'deviation':
            raise RuntimeError(
                "forward_no_endpoints requires 'decode_endpoints=True' and 'mode=\"deviation\"'."
            )
        
        # 1. Decode endpoints from latent vector using numpy-based inversion
        pa, pb = self.forward_endpoints(v)
        pa = pa.to(v.device, non_blocking=True)
        pb = pb.to(v.device, non_blocking=True)

        # 2. Run the main forward pass with the decoded endpoints
        # The main 'forward' method will return (points, lifted_pa, lifted_pb)
        # We only need the points here.
        x, _, _ = self.forward(v, t_b, pa=pa, pb=pb)

        if return_endpoints:
            return x, pa, pb
        return x

    def forward_endpoints(self, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Decodes and inverts endpoint features from a latent vector.

        This is an inference-only helper that uses NumPy to perform the
        computationally intensive inversion of the positional encoding.

        Args:
            v: The latent vector batch, shape (b, embed_dims).

        Returns:
            A tuple containing:
            - Decoded start points `pa` (b, 4) on CPU.
            - Decoded end points `pb` (b, 4) on CPU.
        """
        # 1. Get lifted features from the endpoint head
        lifted_pa_features, lifted_pb_features = self._predict_endpoints_lifted(v)
        lifted_pa_arr = to_array(lifted_pa_features)
        lifted_pb_arr = to_array(lifted_pb_features)

        batch_size = v.shape[0]
        pa = np.zeros((batch_size, 4), dtype=np.float32)
        pb = np.zeros((batch_size, 4), dtype=np.float32)

        # 2. Invert features for each item in the batch
        for i in range(batch_size):
            pa[i] = self._invert_endpoint_features(lifted_pa_arr[i])
            pb[i] = self._invert_endpoint_features(lifted_pb_arr[i])

        return torch.from_numpy(pa), torch.from_numpy(pb)

    def _predict_endpoints_lifted(self, latent_vector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predicts the raw (lifted) endpoint features from the latent vector."""
        head_mode = self.config['head_mode']

        if head_mode == 'separate_relu':
            features = self.mlp_endpoints_relu(latent_vector)
            features = self.mlp_endpoints(features)
        elif head_mode == 'separate':
            # Use the latter part of the latent vector for endpoints
            embed_split = 2 * (self.config['embed_dims'] // 3)
            features = self.mlp_endpoints(latent_vector[:, embed_split:])
        else: # 'default'
            features = self.mlp_endpoints(latent_vector)
        
        # Split into start (pa) and end (pb) features
        lifted_pa, lifted_pb = features.split(self._lifted_endpoint_dim, dim=-1)
        return lifted_pa, lifted_pb

    def _invert_endpoint_features(self, lifted_features: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Inverts the lifted features for a single endpoint to get (pos, rad)."""
        # Invert position (x, y, z)
        pos_features = lifted_features[:self._pos_lifted_dim]
        pos = self.posf_to_pos(self.pos_array, 3, pos_features)

        # Invert radius
        if self.config['rad_mode'] == 'lift':
            # Radius features are the last ones
            rad_features = lifted_features[self._pos_lifted_dim:]
            rad = self.posf_to_pos(self.pos_array, 1, rad_features)
        else: # 'log'
            # Radius is the last single value
            log_rad = lifted_features[-1]
            rad = np.exp(log_rad) - 1.0
            rad = np.array([rad], dtype=np.float32)

        return np.concatenate([pos, rad])

    # --- Static Helper Methods for Coordinate Transformation ---

    @staticmethod
    def make_pos_array_domain(
        grid_size: int,
        wrap_domain: tuple[float, float],
        pos_octaves: list[int]
    ) -> npt.NDArray[np.float32]:
        """Creates a lookup table for inverting positional encoding."""
        inc = 1.0 / (grid_size - 1.0)
        t = np.arange(0.0, 1.0 + inc, inc).astype(np.float32).reshape(-1, 1)
        # Map from [0, 1] to the specified domain
        t = wrap_domain[0] + t * (wrap_domain[1] - wrap_domain[0])
        # Apply positional encoding
        return add_octaves(t, pos_octaves, dim=-1, include_base=False)

    @staticmethod
    def posf_to_pos(
        pos_array: npt.NDArray[np.float32],
        pos_dims: int,
        posf: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """
        Inverts positional features to find original coordinates.

        This function finds the closest match in the pre-computed `pos_array`
        for each dimension of the input features `posf`.

        Args:
            pos_array: The pre-computed lookup table of lifted coordinates.
            pos_dims: The number of spatial dimensions to invert (e.g., 3 for xyz).
            posf: The lifted positional features to invert, shape (d,).

        Returns:
            The inverted coordinates, shape (pos_dims,).
        """
        ret_pos = np.zeros(pos_dims, dtype=np.float32)
        num_octaves_x_lifters = posf.shape[0] // pos_dims
        
        for i in range(pos_dims):
            # Extract features for the current dimension
            start, end = i * num_octaves_x_lifters, (i + 1) * num_octaves_x_lifters
            dim_features = posf[start:end].reshape(1, -1)
            
            # Find the index of the closest entry in the lookup table
            distances = np.linalg.norm(dim_features - pos_array, axis=-1)
            j = np.argmin(distances)

            # Map the index back to the original coordinate space
            grid_size = distances.shape[0]
            d_min, d_max = -0.25, 0.25 # Hardcoded from wrap_domain for simplicity
            ret_pos[i] = d_min + (float(j) / float(grid_size -1)) * (d_max - d_min)

        return ret_pos