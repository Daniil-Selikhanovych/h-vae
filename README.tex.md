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
|[demos/hvae_gaussian_dim_(25/50/100/200/300/400).ipynb](https://github.com/Daniil-Selikhanovych/h-vae/blob/master/demos/hvae_gaussian_dim_300.ipynb) | **HVAE vs PNF/IAF/VB on Gaussian Model:** experiments with learning Hamiltonian Variational Auto-Encoder, Planar Normalizing Flows, Inverse Autoregressive Normalizing Flows, Variational Bayes for Gaussian Model in [https://arxiv.org/abs/1805.11328](https://arxiv.org/abs/1805.11328), comparison of learned $\Delta$ and $\Sigma$ parameters for all methods, comparison of learning processes. Number in the name of notebooks denotes the dimensionality $d$ for the problem.|

For convenience, we have also implemented a framework and located it correspondingly in [gaussians/api](https://github.com/Daniil-Selikhanovych/h-vae/blob/master/gaussians/api).

## Results

### Gaussian Model
We compared all discussed methods for dimensions $d = 25, 50, 100, 200, 300, 400$. Authors trained their models using optimization process for the whole dataset, but we found that HVAE results are better and training process is faster when the dataset is divided on batches. HVAE and normalizing flows were trained for $2000$ iterations across dataset divided on batches with $256$ samples. For all experiments the dataset has $N = 10,000$ points and training was done using RMSProp with a learning rate of $10^{-3}$ and were conducted with fix random seed = $12345$. We average the results for predicted $\theta$ for $3$ different generated datasets according to Gaussian model and present the mean results in the following figures: 
<p align="center">
  <img width="500" alt="Comparison of learned average theta" src="https://github.com/Daniil-Selikhanovych/h-vae/blob/master/images/all_theta_comparison.jpg?raw=true">
</p>

<p align="center">
  <img width="500" alt="Comparison of learned average delta" src="https://github.com/Daniil-Selikhanovych/h-vae/blob/master/images/all_delta_comparison.jpg?raw=true">
</p>

<p align="center">
  <img width="500" alt="Comparison of learned average sigma" src="https://github.com/Daniil-Selikhanovych/h-vae/blob/master/images/all_sigma_comparison.jpg?raw=true">
</p>

<p align="center">
  <img width="500" alt="Comparison of learning processes for different losses" src="https://github.com/Daniil-Selikhanovych/h-vae/blob/master/images/gaussian_learning_processes.jpg?raw=true">
</p>

We trained Variational Bayes for big dimensions $d \geq 100$ more iterations ($3000$ or $4000$) due to the fact that $2000$ iterations were not enough for the convergence of ELBO. HVAE with tempering and IAF have the best learned $\hat{\theta} = \{\Delta, \mathbf{\Sigma}\}$ for the big dimensionality $d \geq 100$. Moreover, HVAE is good for $\mathbf{\Sigma}$ prediction for all dimensions as well Variational Bayes scheme. However, Variationl Bayes suffers most on prediction $\Delta$ as the dimension increases. Planar normalizing flows suffer on prediction $\mathbf{\Sigma}$ compared to IAF. 

Also we compare HVAE with tempering and without tempering, see figure:
<p align="center">
  <img width="500" alt="Comparison across HVAE with and without tempering of learning theta, log-scale" src="https://github.com/Daniil-Selikhanovych/h-vae/blob/master/images/hvae_theta_comparison.jpg?raw=true">
</p>
We can see that the tempered methods perform better than their non-tempered counterparts; this shows that time-inhomogeneous dynamics are a key ingredient in the effectiveness of the method. 

### MNIST
We appeal to the binarized MNIST handwritten digit dataset as an example of image generative task. The training data has the following form: $\mathcal{D}=\left\{x_{1}, \ldots, x_{N}\right\}$, where $x_{i} \in \mathcal{X} \subseteq\{0,1\}^{d}$ for $d=28 \times 28=784$. We then formalize the generative model:
$$
\begin{aligned} z_{i} & \sim \mathcal{N}\left(0, I_{\ell}\right) \\ 
x_{i} \mid z_{i} & \sim \prod_{j=1}^{d} \operatorname{Bernoulli}\left(\left(x_{i}\right)_{j} \mid \pi_{\theta}\left(z_{i}\right)_{j}\right), 
\end{aligned}
$$
for $i \in[N],$ where $\left(x_{i}\right)_{j}$ is the $j^{t h}$ component of $x_{i}, z_{i} \in \mathcal{Z} \equiv \mathbb{R}^{\ell}$  is the latent variable associated with $x_{i},$ and $\pi_{\theta}: \mathcal{Z} \rightarrow \mathcal{X}$ is an encoder (convolutional neural network). The VAE approximate posterior is given by $q_{\theta, \phi}\left(z_{i} \mid x_{i}\right)=\mathcal{N}\left(z_{i} \mid \mu_{\phi}\left(x_{i}\right), \mathbf{\Sigma}_{\phi}\left(x_{i}\right)\right),$ where $\mu_{\phi}$ and $\Sigma_{\phi}$ are separate outputs of the encoder parametrized by $\phi,$ and $\Sigma_{\phi}$ is constrained to be diagonal. 

We set the dimensionalty of the latent space to $l = 64$. As we need both means and variances to parametrize the VAE posterior distribution, the output dimension of the linear layer is set to $l_{\mu, \sigma} = 128$. We use Adam optimizer with standard parameters and learning rate set to $10^{-3}$. 

We compare the performance of HVAE with the performance of IWAE [https://arxiv.org/abs/1509.00519](https://arxiv.org/abs/1509.00519). We set the number of Monte-Carlo steps in HVAE and the number of importance samples in IWAE so that $1$ training epoch requiers equal time to finish. In this setting we can compare them more fairly. We fix the number Monte-Carlo steps $K_{\text{MC}} = 15$ and the number of importance samples $K_{\text{IW}} = 40$. Both models are then optimized for $40$ epochs. Corresponding plots are the following: 
<p align="center">
  <img width="500" alt="The training of HVAE and IWAE models." src="https://github.com/Daniil-Selikhanovych/h-vae/blob/master/images/hvae_iwae_elbo.jpg?raw=true">
</p>

It can be clearly seen that the training loss values are similar for both models, while the validation loss of HVAE is higher due to overfitting. 

It is also important to compare the models outputs in terms of quality. The generated images are shown in the following figures (top - HVAE, bottom - IWAE): 
<p align="center">
  <img width="500" alt="Images generated by both models trained on the binarized MNIST dataset. HVAE." src="https://github.com/Daniil-Selikhanovych/h-vae/blob/master/images/hvae_mnist.png?raw=true">
</p>
<p align="center">
  <img width="500" alt="Images generated by both models trained on the binarized MNIST dataset. IWAE." src="https://github.com/Daniil-Selikhanovych/h-vae/blob/master/images/iwae_mnist.png?raw=true">
</p>
Images generated via IWAE appear to be more blured. However, HVAE tends to generate sharper images while some of them can not be recognized as digits. 

To better understand the behavior of both models, we study the decoded latent vectors of HMC chains (top - HVAE, bottom - IWAE):
<p align="center">
  <img width="500" alt="Trajectories of HMC given a latent vector corresponding to the source image. HVAE." src="https://github.com/Daniil-Selikhanovych/h-vae/blob/master/images/steps_hvae_cropped.png?raw=true">
</p>
<p align="center">
  <img width="500" alt="Trajectories of HMC given a latent vector corresponding to the source image. IWAE." src="https://github.com/Daniil-Selikhanovych/h-vae/blob/master/images/steps_iwae_cropped.png?raw=true">
</p>

In these figures one can clearly see that HVAE encoded vectors often correspond to the class that is different from the ground-truth, even though they are sharper. At the same time, IWAE produces reconstructions that are close to the true images. 

### Details
More details about experiments and settings can be found in the report [https://github.com/Daniil-Selikhanovych/h-vae/blob/master/report/hvae_report.pdf](https://github.com/Daniil-Selikhanovych/h-vae/blob/master/report/hvae_report.pdf).

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
