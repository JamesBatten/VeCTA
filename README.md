# VeCTA: Vessel Centerline Transformer Autoencoder

This repository contains the **official PyTorch implementation** of the *Vessel Centerline Transformer Autoencoder (VeCTA)*—a deep‑learning architecture that learns compact, continuous and **grid‑free** representations of 3‑D vessel segments.

VeCTA is the **Vessel Autoencoder** component in the two‑stage framework presented in our paper:

> **Vector Representations of Vessel Trees**
> James Batten, Michiel Schaap, Matthew Sinclair, Ying Bai, Ben Glocker
> *Medical Imaging with Deep Learning (MIDL), 2025*
> [https://openreview.net/forum?id=ESzOwfBhRv](https://openreview.net/forum?id=ESzOwfBhRv)

VeCTA tokenises complex vessel geometry into a fixed‑length latent vector, enabling downstream tasks such as generative modelling, shape analysis and image‑to‑geometry translation.

---

## Key Features

* **Continuous geometric representation** – grid‑free decoding of a vessel’s 3‑D centre‑line as a continuous function of a 1‑D coordinate, allowing reconstruction at arbitrary resolution.
* **Transformer‑based encoder** – self‑attention processes the sequence of sampled points, making the model robust to variations in shape, length and complexity.
* **Implicit neural representation (INR) decoder** – a neural field conditioned on the latent vector predicts 3‑D coordinates and radius for any point along the vessel.
* **Fourier feature lifting** – sinusoidal embeddings capture high‑frequency geometric details such as curvature and stenosis.
* **Endpoint‑aware architecture** – optional explicit encoding/decoding of the start and end points for accurate reconstruction and seamless graph integration.
* **Variational Autoencoder support** – a single flag turns VeCTA into a VAE, yielding a regularised, semantically smooth latent space for generation and interpolation.

---

## Architecture

VeCTA is an autoencoder composed of two modules: `VesselEncoder` and `VesselDecoder`.

### 1 · Vessel Encoder (`vecta.model.vessel_encoder`)

The encoder compresses a sequence of points sampled from a vessel surface into a fixed‑size latent vector `z_v`.

1. **Input sampling** – a set of `N` points, each `(x, y, z, radius, t)` where `t ∈ [0,1]` is the normalised position along the centre‑line.
2. **Feature lifting** – coordinates are lifted with sinusoidal Fourier features.
3. **Endpoint injection *(optional)* –** lifted start (`p_a`) and end (`p_b`) points are concatenated with point features.
4. **MLP‑A** – projects lifted features to the Transformer working dimension.
5. **Transformer encoder** – multi‑layer self‑attention models point‑wise dependencies.
6. **Aggregation & MLP‑B** – mean‑pool the sequence and pass through an MLP. In VAE mode this splits into mean (`z_μ`) and log‑variance (`z_logσ²`).

### 2 · Vessel Decoder (`vecta.model.vessel_decoder`)

The decoder reconstructs the continuous vessel from `z_v` ("deviation" mode shown):

1. **Inputs** – latent vector `z_v` and a batch of time coordinates `t_b ∈ [0,1]`.
2. **Base interpolation** – a straight line between `p_a` and `p_b`.
3. **Deviation prediction** – an MLP conditioned on `z_v` and lifted `t_b` predicts `(Δx, Δy, Δz, Δr)`.
4. **Modulation** – a function `m(t)` forces the deviation to zero at `t=0` and `t=1` so endpoints match exactly.
5. **Reconstruction** – add the modulated deviation to the base interpolation to obtain the final curve.
6. **Fourier inversion** – an efficient lookup recovers Euclidean endpoints from predicted Fourier features during inference.

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{batten2025_vector,
  author    = {Batten, James and Schaap, Michiel and Sinclair, Matthew and Bai, Ying and Glocker, Ben},
  title     = {Vector Representations of Vessel Trees},
  booktitle = {Proceedings of the 8th Medical Imaging with Deep Learning (MIDL)},
  year      = {2025},
  note      = {Oral presentation},
  url       = {https://openreview.net/forum?id=ESzOwfBhRv}
}
```

---

*Licensed under the Apache 2.0 licence unless stated otherwise.*
