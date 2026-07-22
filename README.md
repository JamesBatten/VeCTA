# VeCTA: Vessel Centerline Transformer Autoencoder

<p align="center">
  <img src="assets/vecta_figure.svg?v=6a11cb0" alt="VECTA figure">
</p>

This repository contains the **official PyTorch implementation** of the *Vessel Centerline Transformer Autoencoder (VeCTA)*—a deep‑learning architecture that learns compact, continuous and **grid‑free** representations of 3‑D vessel segments.

VeCTA is the **Vessel Autoencoder** component in the two‑stage framework presented in our paper:

> **Vector Representations of Vessel Trees**
> James Batten, Michiel Schaap, Matthew Sinclair, Ying Bai, Ben Glocker
> *Medical Imaging with Deep Learning (MIDL), 2025*
> [https://openreview.net/forum?id=ESzOwfBhRv](https://openreview.net/forum?id=ESzOwfBhRv)

VeCTA tokenises complex vessel geometry into a fixed‑length latent vector, enabling downstream tasks such as generative modelling, shape analysis and image‑to‑geometry translation.

Project page: https://jamesbatten.xyz/#project-vessel-centerline-transformer-autoencoder 

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

*Licensed under the [Apache License 2.0](LICENSE) unless stated otherwise.*
