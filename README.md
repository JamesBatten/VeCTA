# VECTA: Vessel Segment Transformer Autoencoder

VECTA is a deep learning model designed to learn meaningful latent representations of individual vessel segments. It operates as an autoencoder, capable of encoding a sequence of points representing a vessel into a compact latent vector and then decoding it back to its original geometric form.

The architecture is built with PyTorch and leverages a Transformer-based encoder to effectively model the sequential relationships between points along a vessel.

## Model Architecture

The model is composed of two primary components: a **`VesselEncoder`** and a **`VesselDecoder`**, integrated within the main `VesselAutoencoder` module.

### 1. Encoder

The Encoder's role is to process a sequence of points that define a vessel segment and compress it into a fixed-size latent vector, `z`.

-   **Input**: A sequence of points representing a vessel segment, where each point has features for 3D position, radius, and time `(x, y, z, r, t)`. It can optionally incorporate the segment's start and end points (`pa`, `pb`) for additional context.
-   **Core Component**: The `VesselEncoder` module, which uses a standard `TransformerEncoder` to process the sequence.
-   **Feature Engineering**:
    -   Input features (position, time, and optionally radius) are "lifted" into a higher-dimensional space using sinusoidal positional encoding (`add_octaves`), which helps the model interpret spatial and temporal information more effectively.
    -   Endpoint information can be concatenated with the sequence features at the beginning of the network or just before the final output layer, controlled by the `inject_endpoints` setting.
-   **Processing**: The lifted feature sequence is first passed through an MLP, then processed by the **`TransformerEncoder`**. This allows the model to capture the global context and relationships between all points in the sequence. The Transformer's output is then aggregated via mean pooling.
-   **Output**: The aggregated vector is mapped to the final latent representation `z` through another MLP. The model can be configured to operate as a standard autoencoder or as a **Variational Autoencoder (VAE)**, in which case it outputs a mean (`z_mu`) and log-variance (`z_logvar`) for the latent distribution.

### 2. Decoder

The Decoder is a conditional generator. It takes the latent vector `z` and a set of time coordinates as input to reconstruct the vessel segment.

-   **Input**:
    1.  The latent vector `z` from the encoder.
    2.  A batch of time coordinates `t_b` at which to generate points.
    3.  Optionally, the ground-truth start and end points (`pa` and `pb`).
-   **Processing**:
    1.  The decoder operates in one of two main modes:
        -   **`default` mode**: Directly predicts the point coordinates `(x, y, z, r)` from the latent vector and encoded time coordinates.
        -   **`deviation` mode**: Predicts a *deviation* from a linear interpolation between the start point `pa` and end point `pb`. This allows the model to focus on learning the complex curvature of the vessel rather than its absolute position, leading to more stable reconstructions.
    2.  For inference in `deviation` mode, the decoder can first predict the endpoints (`pa` and `pb`) directly from the latent vector `z` before proceeding with the full path reconstruction.
-   **Output**: The final output is a sequence of points `(x, y, z, r)` that reconstructs the vessel segment at the specified time coordinates.

## Project Structure

-   `vecta/model/vessel_autoencoder.py`: Contains the main `VesselAutoencoder` module, which integrates the encoder and decoder into a single network.
-   `vecta/model/vessel_encoder.py`: Defines the `VesselEncoder` that processes a vessel point sequence using an MLP and a Transformer stack to produce a latent vector.
-   `vecta/model/vessel_decoder.py`: Defines the `VesselDecoder` that reconstructs a vessel segment from a latent vector and a set of time coordinates, with support for multiple reconstruction modes.
-   `vecta/common/utils.py`: Provides utility functions, most notably `add_octaves` for sinusoidal positional encoding and data type conversions.
-   `vecta/model/{mlp, norm, nonlinearity, weight_init}.py`: A collection of helper modules for building robust and configurable PyTorch network layers in a clean, modular fashion.