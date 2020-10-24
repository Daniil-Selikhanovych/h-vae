# h-vae
Project based on https://arxiv.org/abs/1805.11328



## Requirements

We have run the experiments on Linux. The versions are given in brackets. The following packages are used in the implementation:
* [PyTorch (1.4.0)](https://pytorch.org/get-started/locally/)
* [NumPy (1.17.3)](https://numpy.org/)
* [scikit-learn (0.22.1)](https://scikit-learn.org/stable/)
* [matplotlib (3.1.2)](https://matplotlib.org/)
* [tqdm (4.39.0)](https://github.com/tqdm/tqdm)
* [Pyro (1.3.1)](https://pyro.ai/)
* [TensorFlow (2.1.0) as dependency for GPflow](https://www.tensorflow.org/)


You can use [`pip`](https://pip.pypa.io/en/stable/) or [`conda`](https://docs.conda.io/en/latest/) to install them. 

## Contents

All the experiments can be found in the underlying notebooks:

| Notebook      | Description |
|-----------|------------|
|[demos/mnist.ipynb](https://github.com/Daniil-Selikhanovych/h-vae/blob/master/demos/mnist.ipynb) | **HVAE vs IWAE on MNIST:** experiments with HMC, training of the Hamiltonian Variational Auto-Encoder and Importance Weighted Autoencoder, reconstruction of encoded images, comparison of HMC trajectories.|

For convenience, we have also implemented a framework and located it correspondingly in [gaussians/api](https://github.com/Daniil-Selikhanovych/h-vae/blob/master/gaussians/api).

## Our team

At the moment we are *Skoltech DS MSc, 2019-2021* students.
* Artemenkov Aleksandr 
* Karpikov Igor
* Selikhanovych Daniil