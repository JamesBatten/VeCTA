from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Literal, Optional, Union

import torch
import torch.nn as nn

from vecta.common.utils import add_octaves
from vecta.model.mlp import MLP2


@dataclass
class VesselEncoderConfig:
    """Configuration for the VesselEncoder model."""
    sa_dim: int = 5  # Input samples feature dim (x, y, z, r, t)
    vdim: int = 64
    n_heads: int = 8
    mlp_hidden_dims: int = 2048
    input_activation: str = 'gelu'
    mlp_b_activation: str = 'gelu'
    mlp_b_norm: Optional[str] = None
    activation_transformer: str = 'relu'
    dim_feedforward_transformer: int = 2048
    n_transformer_layers: int = 3
    dropout: float = 0.0
    octaves: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    output_dims: int = 64
    encode_endpoints: bool = False
    rad_mode: Literal['lift', 'log'] = 'lift'
    inject_endpoints: Literal['start', 'end'] = 'start'
    use_vae: bool = False

    def __post_init__(self):
        """Validate configuration values after initialization."""
        if self.rad_mode not in ['lift', 'log']:
            raise ValueError(f"rad_mode must be 'lift' or 'log', not '{self.rad_mode}'")
        if self.inject_endpoints not in ['start', 'end']:
            raise ValueError(f"inject_endpoints must be 'start' or 'end', not '{self.inject_endpoints}'")


class VesselEncoder(nn.Module):
    """
    Encodes a vessel representation into a fixed-size latent vector.

    This module uses a Transformer-based architecture to process a sequence of
    points (x, y, z, radius, time) representing a vessel segment. It can
    optionally incorporate the vessel's start and end points into the encoding
    process. The encoder can function as a standard autoencoder bottleneck or
    as the encoder part of a Variational Autoencoder (VAE).

    The input features are first positionally encoded ("lifted") using
    sinusoidal functions before being passed to an MLP and then a Transformer
    stack. The aggregated output from the Transformer is then processed by a
    final MLP to produce the latent vector.
    """

    def __init__(self, config: Optional[Union[dict, VesselEncoderConfig]] = None):
        """
        Initializes the VesselEncoder module.

        Args:
            config: A configuration dictionary or a VesselEncoderConfig object.
                    If None, default configuration is used.
        """
        super().__init__()

        if config is None:
            self.config = VesselEncoderConfig()
        elif isinstance(config, dict):
            self.config = VesselEncoderConfig(**config)
        else:
            self.config = config

        ff_dim = self.config.vdim * self.config.n_heads

        # --- Calculate input dimensions for MLPs ---
        # Dimension of the main sequence features after processing.
        mlp_a_in_dims = self._get_processed_feature_dim(
            num_pos_dims=3, num_rad_dims=1, num_time_dims=1
        )
        
        # Dimension of endpoint features after processing (pa and pb).
        endpoint_dims = 0
        if self.config.encode_endpoints:
            # Each endpoint has (pos, rad), so 2 endpoints * features_per_endpoint
            endpoint_dims = 2 * self._get_processed_feature_dim(
                num_pos_dims=3, num_rad_dims=1, num_time_dims=0
            )

        if self.config.encode_endpoints and self.config.inject_endpoints == 'start':
            mlp_a_in_dims += endpoint_dims

        # --- Define neural network layers ---
        self.mlp_a = MLP2(
            mlp_a_in_dims, self.config.mlp_hidden_dims,
            ff_dim, self.config.input_activation
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=ff_dim,
            nhead=self.config.n_heads,
            dim_feedforward=self.config.dim_feedforward_transformer,
            dropout=self.config.dropout,
            activation=self.config.activation_transformer,
            batch_first=True  # Use batch_first for intuitive tensor shapes
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=self.config.n_transformer_layers
        )

        # Dimension for the final MLP (mlp_b)
        mlp_b_in_dims = ff_dim
        if self.config.encode_endpoints and self.config.inject_endpoints == 'end':
            mlp_b_in_dims += endpoint_dims

        if not self.config.use_vae:
            self.mlp_b = MLP2(
                mlp_b_in_dims, self.config.mlp_hidden_dims,
                self.config.output_dims, self.config.mlp_b_activation,
                norm_1=self.config.mlp_b_norm
            )
        else:
            # VAE outputs for mean and log-variance
            self.mlp_b_mu = MLP2(
                mlp_b_in_dims, self.config.mlp_hidden_dims,
                self.config.output_dims, self.config.mlp_b_activation,
                norm_1=self.config.mlp_b_norm
            )
            self.mlp_b_logvar = MLP2(
                mlp_b_in_dims, self.config.mlp_hidden_dims,
                self.config.output_dims, self.config.mlp_b_activation,
                norm_1=self.config.mlp_b_norm
            )

    @classmethod
    def default_config(cls) -> dict:
        """Returns the default configuration as a dictionary."""
        return asdict(VesselEncoderConfig())

    def _get_processed_feature_dim(self, num_pos_dims: int, num_rad_dims: int, num_time_dims: int) -> int:
        """Calculates the total dimension of features after processing."""
        num_octaves = len(self.config.octaves)
        LIFTERS_COUNT = 2  # sin and cos

        # Dimensions for features that are always positionally encoded
        lifted_dims = (num_pos_dims + num_time_dims) * LIFTERS_COUNT * num_octaves

        # Dimensions for radius feature, which depends on rad_mode
        if self.config.rad_mode == 'lift':
            rad_dim = num_rad_dims * LIFTERS_COUNT * num_octaves
        else:  # 'log' mode
            rad_dim = num_rad_dims
            
        return lifted_dims + rad_dim

    def _process_features(
        self, pos: torch.Tensor, rad: torch.Tensor, t: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Applies positional encoding and transformation to input features."""
        # Process position (always lifted)
        pos_lifted = add_octaves(pos, self.config.octaves, dim=-1, include_base=False)
        
        # Process radius (lifted or log-transformed)
        if self.config.rad_mode == 'log':
            # Ensure radius is non-negative before log
            rad_processed = torch.log(1.0 + rad.clamp(min=0.0)).float()
        else:  # 'lift'
            rad_processed = add_octaves(rad, self.config.octaves, dim=-1, include_base=False)

        # Process time if provided (always lifted)
        if t is not None:
            t_lifted = add_octaves(t, self.config.octaves, dim=-1, include_base=False)
            return torch.cat([pos_lifted, rad_processed, t_lifted], dim=-1)
        
        return torch.cat([pos_lifted, rad_processed], dim=-1)

    def forward(
        self, x: torch.Tensor, pa: Optional[torch.Tensor] = None, pb: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Performs the forward pass of the encoder.

        Args:
            x: Input tensor of shape (b, n, sa_dim), where sa_dim corresponds to
               (x, y, z, r, t).
            pa: Start point tensor of shape (b, 4), corresponding to (x, y, z, r).
            pb: End point tensor of shape (b, 4), corresponding to (x, y, z, r).

        Returns:
            - If use_vae is False: a latent tensor of shape (b, output_dims).
            - If use_vae is True: a tuple (z, z_mu, z_logvar), where z is the
              reparameterized sample.
        """
        # 1. Process main sequence features
        # x: (b, n, 5) -> pos:(b,n,3), rad:(b,n,1), t:(b,n,1)
        x_pos, x_rad, t_t = x.split([3, 1, 1], dim=-1)
        processed_x = self._process_features(x_pos, x_rad, t_t) # (b, n, processed_dim)
        
        # 2. Process and inject endpoint features (if configured)
        processed_endpoints = None
        if self.config.encode_endpoints:
            if pa is None or pb is None:
                raise ValueError("Endpoints 'pa' and 'pb' must be provided when encode_endpoints is True.")
            
            # pa/pb: (b, 4) -> pos:(b,3), rad:(b,1)
            pa_pos, pa_rad = pa.split([3, 1], dim=-1)
            pb_pos, pb_rad = pb.split([3, 1], dim=-1)

            processed_pa = self._process_features(pa_pos, pa_rad) # (b, processed_endpoint_dim)
            processed_pb = self._process_features(pb_pos, pb_rad) # (b, processed_endpoint_dim)
            processed_endpoints = torch.cat([processed_pa, processed_pb], dim=-1) # (b, 2 * processed_endpoint_dim)

            if self.config.inject_endpoints == 'start':
                # Expand endpoints to match sequence length and concatenate
                endpoints_expanded = processed_endpoints.unsqueeze(1).expand(-1, x.size(1), -1)
                processed_x = torch.cat([processed_x, endpoints_expanded], dim=-1)

        # 3. Pass through MLP-A and Transformer
        x = self.mlp_a(processed_x)           # (b, n, ff_dim)
        x = self.transformer(x)              # (b, n, ff_dim)
        x = torch.mean(x, dim=1)             # (b, ff_dim), aggregate sequence

        # 4. Inject endpoints at the end if configured
        if self.config.encode_endpoints and self.config.inject_endpoints == 'end':
            x = torch.cat([x, processed_endpoints], dim=-1)

        # 5. Pass through MLP-B (or VAE heads)
        if not self.config.use_vae:
            return self.mlp_b(x)

        z_mu = self.mlp_b_mu(x)
        z_logvar = self.mlp_b_logvar(x)
        z = self.reparametrize(z_mu, z_logvar)
        return z, z_mu, z_logvar

    def reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for VAE.
        
        z = mu + std * epsilon
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std