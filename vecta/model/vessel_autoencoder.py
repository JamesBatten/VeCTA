from __future__ import annotations

from typing import Any, Optional, Union

import torch
import torch.nn as nn

from vecta.model.vessel_decoder import VesselDecoder
from vecta.model.vessel_encoder import VesselEncoder, VesselEncoderConfig


class VesselAutoencoder(nn.Module):
    """
    A complete autoencoder for learning representations of vessel segments.

    This module encapsulates a `VesselEncoder` and a `VesselDecoder` to form a
    full autoencoder architecture. It takes a sequence of points representing a
    vessel, encodes it into a fixed-size latent vector, and then decodes it
    back into a sequence of points.

    The behavior of the encoder and decoder is controlled by a single, unified
    configuration dictionary provided during initialization. This class is
    responsible for correctly mapping these high-level settings to the specific
    configurations of its submodules.

    The model can operate as a standard autoencoder or as a Variational
    Autoencoder (VAE) by setting the `use_vae` flag in the configuration.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None):
        """
        Initializes the VesselAutoencoder.

        Args:
            config: A dictionary of configuration parameters. If None, the
                    default configuration is used.
        """
        super().__init__()
        if config is None:
            config = self.default_config()
        self.config = config

        # Prepare configurations for submodules and instantiate them
        encoder_config = self._prepare_encoder_config()
        self.venc = VesselEncoder(encoder_config)

        decoder_config = self._prepare_decoder_config()
        self.vdec = VesselDecoder(decoder_config)

    def _prepare_encoder_config(self) -> VesselEncoderConfig:
        """Creates the VesselEncoderConfig from the main config dictionary."""
        # Note: The original code set a generic 'activation' key, which is not
        # used by VesselEncoderConfig. We map 'encoder_activation' to
        # 'input_activation' as it's the most likely intended behavior.
        return VesselEncoderConfig(
            n_heads=self.config['n_heads'],
            dim_feedforward_transformer=self.config['dim_feedforward_transformer'],
            n_transformer_layers=self.config['encoder_layers'],
            mlp_hidden_dims=self.config['encoder_hidden_dims'],
            input_activation=self.config['encoder_activation'],
            output_dims=self.config['bottleneck_dim'],
            octaves=self.config['octaves'],
            rad_mode=self.config['encoder_rad_mode'],
            encode_endpoints=self.config['include_endpoints'],
            use_vae=self.config['use_vae'],
            inject_endpoints=self.config['encoder_inject_endpoints'],
            mlp_b_activation=self.config['encoder_mlp_b_activation'],
            mlp_b_norm=self.config['encoder_mlp_b_norm'],
        )

    def _prepare_decoder_config(self) -> dict[str, Any]:
        """Creates the VesselDecoder configuration from the main config."""
        # Start with decoder defaults and override with autoencoder settings.
        vdec_config = VesselDecoder.default_config()
        vdec_config.update({
            'in_dims': 2 * len(self.config['octaves']) + self.config['bottleneck_dim'],
            'embed_dims': self.config['bottleneck_dim'],
            'hidden_dims': self.config['decoder_hidden_dims'],
            'n_layers': self.config['decoder_layers'],
            'activation': self.config['decoder_activation'],
            'mode': self.config['decoder_mode'],
            'modulation': self.config['modulation'],
            'octaves': self.config['octaves'],
            'rad_mode': self.config['decoder_rad_mode'],
            'head_mode': self.config['decoder_head_mode'],
            'decode_endpoints': self.config['include_endpoints'],
        })
        return vdec_config

    @classmethod
    def default_config(cls) -> dict[str, Any]:
            """Provides the default configuration for the autoencoder."""
            return {
                'n_heads': 8,
                'dim_feedforward_transformer': 2048,
                'bottleneck_dim': 64,
                'encoder_layers': 3,
                'encoder_hidden_dims': 2048,
                'decoder_hidden_dims': 2048,
                'encoder_activation': 'gelu',
                'decoder_activation': 'relu',
                'decoder_mode': 'deviation',
                'decoder_layers': 2,
                'modulation': 'default',  # 'default' or 'none'
                'octaves': [1, 2, 4, 8, 16, 32],
                'include_endpoints': False,
                'encoder_inject_endpoints': 'start',
                'encoder_mlp_b_activation': 'gelu',
                'encoder_mlp_b_norm': None,
                'use_vae': False,
                'encoder_rad_mode': 'lift',
                'decoder_rad_mode': 'log',
                'decoder_head_mode': 'default'
            }

    def forward(
        self,
        input_sequence: torch.Tensor,
        target_times: torch.Tensor,
        pa: Optional[torch.Tensor] = None,
        pb: Optional[torch.Tensor] = None,
        noise_stddev: Optional[float] = None,
        return_mulogvar: bool = False
    ) -> Union[torch.Tensor, tuple]:
        """
        Performs the full autoencoder forward pass.

        Args:
            input_sequence: The input tensor of vessel points, shape (b, n, 5).
            target_times: The time coordinates for the decoder, shape (b, k).
            pa: The start points tensor, shape (b, 4).
            pb: The end points tensor, shape (b, 4).
            noise_stddev: If provided, adds Gaussian noise with this standard
                          deviation to the latent vector.
            return_mulogvar: If True (and `use_vae` is True), returns the VAE's
                             mean and log-variance along with the decoded output.

        Returns:
            - The decoded output from the `VesselDecoder`.
            - If `return_mulogvar` is True, a tuple containing the decoded
              output and the VAE parameters (z_mu, z_logvar). The structure of
              the decoded output depends on the decoder's configuration.
        """
        # 1. Encode the input sequence into a latent representation.
        z_mu, z_logvar = None, None
        if not self.config['use_vae']:
            latent_vector = self.venc(input_sequence, pa=pa, pb=pb)
        else:
            latent_vector, z_mu, z_logvar = self.venc(input_sequence, pa=pa, pb=pb)

        # 2. Optionally inject noise into the latent space for regularization.
        if noise_stddev is not None:
            assert isinstance(noise_stddev, float), "noise_stddev must be a float."
            noise = torch.randn_like(latent_vector) * noise_stddev
            latent_vector += noise

        # 3. Decode the latent vector to reconstruct the vessel.
        decoded_output = self.vdec(latent_vector, target_times, pa=pa, pb=pb)

        # 4. Return the appropriate outputs based on the mode.
        if return_mulogvar:
            if not self.config['use_vae']:
                raise ValueError("`return_mulogvar=True` is only valid when `use_vae=True`.")
            # When this flag is true, the decoder output is expected to be a
            # tuple (out, out_pa, out_pb) if `include_endpoints` is true.
            out, out_pa, out_pb = decoded_output
            return out, out_pa, out_pb, z_mu, z_logvar

        return decoded_output