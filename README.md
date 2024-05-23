<div align="center">

<img src="http://drive.google.com/uc?export=view&id=1jMpe_5_XZpozJviP7BNO9k1VuSOyfhxR" width="400">
</br>
</br>

<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rezaakb/pinns-jax/blob/main/tutorials/0-Burgers.ipynb)

<a href="https://arxiv.org/abs/2311.03626">[Paper]</a> - <a href="https://github.com/rezaakb/pinns-torch">[PyTorch]</a> - <a href="https://github.com/rezaakb/pinns-tf2">[TensorFlow v2]</a> - <a href="https://github.com/maziarraissi/PINNs">[TensorFlow v1]</a>
</div>

## Description

This paper presents ‘PINNs-JAX’, an innovative implementation that utilizes the JAX framework to leverage the distinct capabilities of XLA compilers. This approach aims to improve computational efficiency and flexibility within PINN applications.

<div align="center">
<img src="http://drive.google.com/uc?export=view&id=1bhiyum1xh2KnLOnMeTjevBgOA8m4Qkel" width="1000">
</br>
<em>Each subplot corresponds to a problem, with its iteration count displayed at the
top. The logarithmic x-axis shows the speed-up factor w.r.t the original code in TensorFlow v1, and the y-axis illustrates the mean relative error.</em>
</div>
</br>


For more information, please refer to our paper:

<a href="https://openreview.net/pdf?id=BPFzolSSrI">Comparing PINNs Across Frameworks: JAX, TensorFlow, and PyTorch.</a> Reza Akbarian Bafghi, and Maziar Raissi. AI4DiffEqtnsInSci, ICLR, 2024.

## Installation

PINNs-JAX requires following dependencies to be installed:

- [JAX](https://jax.readthedocs.io/en/latest/installation.html) >= 0.4.16
- [Hydra](https://hydra.cc/docs/intro/) >= 1.3

Then, you can install PINNs-JAX itself via \[pip\]:

```bash
pip install pinnsjax
```

If you intend to introduce new functionalities or make code modifications, we suggest duplicating the repository and setting up a local installation:

```bash
git clone https://github.com/rezaakb/pinns-jax
cd pinns-jax

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install package
pip install -e .
```

## Quick start

Explore a variety of implemented examples within the [examples](examples) folder. To run a specific code, such as the one for the Navier-Stokes PDE, you can use:

```bash
python examples/navier_stokes/train.py
```

You can train the model using a specified configuration, like the one found in [examples/navier_stokes/configs/config.yaml](examples/navier_stokes/configs/config.yaml). Parameters can be overridden directly from the command line. For instance:

```bash
python examples/navier_stokes/train.py trainer.max_epochs=20 n_train=3000
```

To utilize our package, there are two primary options:

- Implement your training structures using Hydra, as illustrated in our provided examples.
- Directly incorporate our package to solve your custom problem.

For a practical guide on directly using our package to solve the Schrödinger PDE in a continuous forward problem, refer to our tutorial here: [tutorials/0-Burgers.ipynb](tutorials/0-Burgers.ipynb).

## Data

The data located on the server and will be downloaded automatically upon running each example.

## Contributing

As this is the first version of our package, there might be scope for enhancements and bug fixes. We highly value community contributions. If you find any issues, missing features, or unusual behavior during your usage of this library, please feel free to open an issue or submit a pull request on GitHub. For any queries, suggestions, or feedback, please send them to [Reza Akbarian Bafghi](https://www.linkedin.com/in/rezaakbarian/) at [reza.akbarianbafghi@colorado.edu](mailto:reza.akbarianbafghi@colorado.edu).

## License

Distributed under the terms of the \[BSD-3\] license, "pinnsjax" is free and open source software.

## Resources

We employed [this template](https://github.com/ashleve/lightning-hydra-template) to develop the package, drawing from its structure and design principles. For a deeper understanding, we recommend visiting their GitHub repository.

## Citation

```
@inproceedings{
bafghi2024comparing,
title={Comparing {PINN}s Across Frameworks: {JAX}, TensorFlow, and PyTorch},
author={Reza Akbarian Bafghi and Maziar Raissi},
booktitle={ICLR 2024 Workshop on AI4DifferentialEquations In Science},
year={2024},
url={https://openreview.net/forum?id=BPFzolSSrI}
}
```
