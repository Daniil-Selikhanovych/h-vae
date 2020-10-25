# h-vae
Project based on [https://arxiv.org/abs/1805.11328](https://arxiv.org/abs/1805.11328).

## Used papers
Importance Weighted Autoencoder - [https://arxiv.org/abs/1509.00519](https://arxiv.org/abs/1509.00519);
Planar Normalizing Flows - [https://arxiv.org/abs/1505.05770](https://arxiv.org/abs/1505.05770);
Inverse Autoregressive Flows - [https://arxiv.org/abs/1606.04934](https://arxiv.org/abs/1606.04934).

## Project proposal
Link to the project proposal - [https://drive.google.com/file/d/1-q50kvccrze68GvE1DEoaRx2Kq_54LeG/view?usp=sharing](https://drive.google.com/file/d/1-q50kvccrze68GvE1DEoaRx2Kq_54LeG/view?usp=sharing).

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

All experiments can be found in the underlying notebooks:

| Notebook      | Description |
|-----------|------------|
|[demos/mnist.ipynb](https://github.com/Daniil-Selikhanovych/h-vae/blob/master/demos/mnist.ipynb) | **HVAE vs IWAE on MNIST:** experiments with HMC, training of the Hamiltonian Variational Auto-Encoder and Importance Weighted Autoencoder, reconstruction of encoded images, comparison of HMC trajectories.|
|[demos/hvae_gaussian_dim_(25/50/100/200/300/400).ipynb](https://github.com/Daniil-Selikhanovych/h-vae/blob/master/demos/hvae_gaussian_dim_300.ipynb) | **HVAE vs PNF/IAF/VB on Gaussian Model:** experiments with learning Hamiltonian Variational Auto-Encoder, Planar Normalizing Flows, Inverse Autoregressive Normalizing Flows, Variational Bayes for Gaussian Model in [https://arxiv.org/abs/1805.11328](https://arxiv.org/abs/1805.11328), comparison of learned <img src="svgs/7e9fe18dc67705c858c077c5ee292ab4.svg?invert_in_darkmode" align=middle width=13.69867124999999pt height=22.465723500000017pt/> and <img src="svgs/813cd865c037c89fcdc609b25c465a05.svg?invert_in_darkmode" align=middle width=11.87217899999999pt height=22.465723500000017pt/> parameters for all methods, comparison of learning processes. Number in the name of notebooks denotes the dimensionality <img src="svgs/2103f85b8b1477f430fc407cad462224.svg?invert_in_darkmode" align=middle width=8.55596444999999pt height=22.831056599999986pt/> for the problem.|

For convenience, we have also implemented a framework and located it correspondingly in [gaussians/api](https://github.com/Daniil-Selikhanovych/h-vae/blob/master/gaussians/api).

## Results
We compared all discussed methods for dimensions <img src="svgs/f67096da04471c9f50e31b00f7f50c14.svg?invert_in_darkmode" align=middle width=198.5103615pt height=22.831056599999986pt/>. Authors trained their models using optimization process for the whole dataset, but we found that HVAE results are better and training process is faster when the dataset is divided on batches. HVAE and normalizing flows were trained for <img src="svgs/946a7aaf620371ac3590184a18ac92c1.svg?invert_in_darkmode" align=middle width=32.876837399999985pt height=21.18721440000001pt/> iterations across dataset divided on batches with <img src="svgs/9684129ebb778f48019391de80875252.svg?invert_in_darkmode" align=middle width=24.657628049999992pt height=21.18721440000001pt/> samples. For all experiments the dataset has <img src="svgs/28326d3ee086205259a55f1263e21783.svg?invert_in_darkmode" align=middle width=85.31952989999999pt height=22.465723500000017pt/> points and training was done using RMSProp with a learning rate of <img src="svgs/7478f3ddcc5c4a0d602772a3057efe42.svg?invert_in_darkmode" align=middle width=33.26498669999999pt height=26.76175259999998pt/> and were conducted with fix random seed = <img src="svgs/66598bc181ac25cca9c745e3ed395aec.svg?invert_in_darkmode" align=middle width=41.09604674999999pt height=21.18721440000001pt/>. We average the results for predicted <img src="svgs/27e556cf3caa0673ac49a8f0de3c73ca.svg?invert_in_darkmode" align=middle width=8.17352744999999pt height=22.831056599999986pt/> for <img src="svgs/5dc642f297e291cfdde8982599601d7e.svg?invert_in_darkmode" align=middle width=8.219209349999991pt height=21.18721440000001pt/> different generated datasets according to Gaussian model and present the mean results in the following figures: 
<p align="center">
  <img width="500" alt="Comparison of averages of <img src="svgs/5599bf1a72afc20a8fee2155ca64725c.svg?invert_in_darkmode" align=middle width=85.22671409999998pt height=31.50689519999998pt/> for several variational methods and choices of dimensionality <img src="svgs/2103f85b8b1477f430fc407cad462224.svg?invert_in_darkmode" align=middle width=8.55596444999999pt height=22.831056599999986pt/>" src="https://github.com/Daniil-Selikhanovych/h-vae/blob/master/images/all_theta_comparison.jpg?raw=true">
</p>

<p align="center">
  <img width="500" alt="Comparison of averages of <img src="svgs/a0f395e5fc3b7094811899b1cd9554ff.svg?invert_in_darkmode" align=middle width=96.73363919999998pt height=31.141535699999984pt/> for several variational methods and choices of dimensionality <img src="svgs/2103f85b8b1477f430fc407cad462224.svg?invert_in_darkmode" align=middle width=8.55596444999999pt height=22.831056599999986pt/>" src="https://github.com/Daniil-Selikhanovych/h-vae/blob/master/images/all_delta_comparison.jpg?raw=true">
</p>

<p align="center">
  <img width="500" alt="Comparison of averages of <img src="svgs/ebeaa8b45891626fc6a4a8e1fd6cb53f.svg?invert_in_darkmode" align=middle width=96.64208729999999pt height=31.23293909999999pt/> for several variational methods and choices of dimensionality <img src="svgs/2103f85b8b1477f430fc407cad462224.svg?invert_in_darkmode" align=middle width=8.55596444999999pt height=22.831056599999986pt/>" src="https://github.com/Daniil-Selikhanovych/h-vae/blob/master/images/all_delta_comparison.jpg?raw=true">
</p>

<p align="center">
  <img width="500" alt="Comparison of learning processes for different losses" src="https://github.com/Daniil-Selikhanovych/h-vae/blob/master/images/gaussian_learning_processes.jpg?raw=true">
</p>
For Variational Bayes we trained it for big dimensionality <img src="svgs/c822ed4d73bf20944683878604a747dd.svg?invert_in_darkmode" align=middle width=55.13122229999999pt height=22.831056599999986pt/> more iterations (<img src="svgs/124a322b51f8b8228ecd83a0a70c995d.svg?invert_in_darkmode" align=middle width=32.876837399999985pt height=21.18721440000001pt/> or <img src="svgs/fcaa39039914d95fd04b3a004da93bf2.svg?invert_in_darkmode" align=middle width=32.876837399999985pt height=21.18721440000001pt/>) due to the fact that <img src="svgs/946a7aaf620371ac3590184a18ac92c1.svg?invert_in_darkmode" align=middle width=32.876837399999985pt height=21.18721440000001pt/> iterations were not enough for the convergence of ELBO. HVAE with tempering and IAF have the best learned <img src="svgs/494429435f09e010cd0f9ba62ffc7c59.svg?invert_in_darkmode" align=middle width=81.18702734999998pt height=31.50689519999998pt/> for the big dimensionality <img src="svgs/c822ed4d73bf20944683878604a747dd.svg?invert_in_darkmode" align=middle width=55.13122229999999pt height=22.831056599999986pt/>. Moreover, HVAE is good for <img src="svgs/c5f57ca07cdaed981ff201596a66a1b3.svg?invert_in_darkmode" align=middle width=13.652895299999988pt height=22.55708729999998pt/> prediction for all dimensions as well Variational Bayes scheme. However, Variationl Bayes suffers most on prediction <img src="svgs/7e9fe18dc67705c858c077c5ee292ab4.svg?invert_in_darkmode" align=middle width=13.69867124999999pt height=22.465723500000017pt/> as the dimension increases. Planar normalizing flows suffer on prediction <img src="svgs/c5f57ca07cdaed981ff201596a66a1b3.svg?invert_in_darkmode" align=middle width=13.652895299999988pt height=22.55708729999998pt/> compared to IAF. 

Also we compare HVAE with tempering and without tempering, see figure:
<p align="center">
  <img width="500" alt="Comparison across HVAE with and without tempering of learning <img src="svgs/27e556cf3caa0673ac49a8f0de3c73ca.svg?invert_in_darkmode" align=middle width=8.17352744999999pt height=22.831056599999986pt/>, log-scale" src="https://github.com/Daniil-Selikhanovych/h-vae/blob/master/images/hvae_theta_comparison.jpg?raw=true">
</p>
We can see that the tempered methods perform better than their non-tempered counterparts; this shows that time-inhomogeneous dynamics are a key ingredient in the effectiveness of the method. 

## Pretrained Models

| Models      | Description |
|-----------|------------|
|[HVAE & IWAE on MNIST](https://drive.google.com/drive/folders/18KuruFMjmGfgyt_km747P4QuDYtVbJec?usp=sharing) |models from experiments in [demos/mnist.ipynb](https://github.com/Daniil-Selikhanovych/h-vae/blob/master/demos/mnist.ipynb)|

## LaTex in Readme

We have used [`readme2tex`](https://github.com/leegao/readme2tex) to render LaTex code in this Readme. Install the corresponding hook and change the command to fix the issue with broken paths:
```bash
python -m readme2tex --output README.md README.tex.md  --branch master --svgdir 'svgs' --nocdn
```


## Our team

At the moment we are *Skoltech DS MSc, 2019-2021* students.
* Artemenkov Aleksandr 
* Karpikov Igor
* Selikhanovych Daniil
