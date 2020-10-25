# h-vae
Project based on https://arxiv.org/abs/1805.11328



## Requirements

We have run the experiments on Linux. The versions are given in brackets. The following packages are used in the implementation:
* [PyTorch (1.4.0)](https://pytorch.org/get-started/locally/)
* [NumPy (1.17.3)](https://numpy.org/)
* [SciPy (1.5.3)](https://docs.scipy.org/doc/)
* [scikit-learn (0.22.1)](https://scikit-learn.org/stable/)
* [matplotlib (3.1.2)](https://matplotlib.org/)
* [tqdm (4.39.0)](https://github.com/tqdm/tqdm)
* [Pyro (1.3.1)](https://pyro.ai/)


You can use [`pip`](https://pip.pypa.io/en/stable/) or [`conda`](https://docs.conda.io/en/latest/) to install them. 

## Contents

All the experiments can be found in the underlying notebooks:

| Notebook      | Description |
|-----------|------------|
|[demos/mnist.ipynb](https://github.com/Daniil-Selikhanovych/h-vae/blob/master/demos/mnist.ipynb) | **HVAE vs IWAE on MNIST:** experiments with HMC, training of the Hamiltonian Variational Auto-Encoder and Importance Weighted Autoencoder, reconstruction of encoded images, comparison of HMC trajectories.|
|[demos/hvae_gaussian_dim_(25/50/100/200/300/400).ipynb](https://github.com/Daniil-Selikhanovych/h-vae/blob/master/demos/hvae_gaussian_dim_300.ipynb) | **HVAE vs PNF/IAF/VB on Gaussian Model:** experiments with learning Hamiltonian Variational Auto-Encoder, Planar Normalizing Flows, Inverse Autoregressive Normalizing Flows, Variational Bayes for Gaussian Model in [https://arxiv.org/abs/1805.11328](https://arxiv.org/abs/1805.11328), comparison of learned <img src="https://rawgit.com/in	git@github.com:Daniil-Selikhanovych/h-vae/svgs/svgs/7e9fe18dc67705c858c077c5ee292ab4.svg?invert_in_darkmode" align=middle width=13.69867124999999pt height=22.465723500000017pt/> and sigma parameters for all methods, comparison of learning processes. Number in the name of notebooks denotes the dimensionality <img src="https://rawgit.com/in	git@github.com:Daniil-Selikhanovych/h-vae/svgs/svgs/2103f85b8b1477f430fc407cad462224.svg?invert_in_darkmode" align=middle width=8.55596444999999pt height=22.831056599999986pt/> for the problem.|

For convenience, we have also implemented a framework and located it correspondingly in [gaussians/api](https://github.com/Daniil-Selikhanovych/h-vae/blob/master/gaussians/api).

## Our team

At the moment we are *Skoltech DS MSc, 2019-2021* students.
* Artemenkov Aleksandr 
* Karpikov Igor
* Selikhanovych Daniil
