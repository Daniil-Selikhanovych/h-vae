import numpy as np

import torch
from torch.distributions.normal import Normal
import torch.optim as optim
from torch.autograd import Variable

import time
import pickle
import os

class BaseModel():
    """ Base model containing generic methods for running the tests """

    def __init__(self, params, var_names, var_inits, model_name, d):
        """ 
        Initialize model from variable names and initial values 
        
        Args
            params: dict of parameters for setup
            var_names: PyTorch variable names
            var_inits: Initial values for variables as list of PyTorch arrays
            model_name: Name of the model to test
            d: Dimensionality of the problem
        """

        # Variable initialization
        self.dtype = var_inits[0].dtype
        self.device = var_inits[0].device
        
        torch_vars = []
        for i in range(len(var_inits)):
            torch_vars.append(Variable(var_inits[i], 
                                       requires_grad=True).to(self.device))

        self.params = params
        self.var_names = var_names
        self.torch_vars = torch_vars
        self.var_inits = var_inits
        self.model_name = model_name

        self.d = d
        
        self.n_batch = params['n_batch']
        self.std_norm = Normal(loc=torch.zeros(d, dtype=self.dtype), 
                               scale=torch.ones(d, dtype=self.dtype))

        # First two PyTorch variables will be the same across methods
        self.delta = torch_vars[0]
        self.log_sigma = torch_vars[1]

        # Filepath to save results
        self.save_dir = os.path.join('save', str(self.d))
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
    def train(self, train_x, train_ind):
        """ 
        Train the model.
        Args:
            train_x: PyTorch training data
            train_ind: Index of training run
        Returns:
            delta: Value of delta at end of training
            sigma: Value of sigma at end of training
        The history of the run, given by the hist_dict variable, is also 
        saved by pickle to the file 
            ./save/<d>/<self.method_name>_train_<train_ind>.p
        """      

        # Start by initializing parameter and ELBO history
        hist_length = int(self.params['n_iter']/self.params['save_every']) + 1
        hist_dict = {}

        for j in range(len(self.torch_vars)):
            initial = self.var_inits[j].clone().detach().cpu()

            # Careful with scalars vs. arrays
            if len(initial.shape) > 0:
                hist_array = torch.zeros([hist_length] + list(initial.shape), 
                                         dtype=self.dtype).numpy()
                hist_array[0, :] = initial
            else:
                hist_array = torch.zeros(hist_length, dtype=self.dtype).numpy()
                hist_array[0] = initial

            hist_dict[self.var_names[j]] = hist_array
        
        self.x_bar = torch.sum(train_x, 0)
        self.C_xx = torch.einsum('ij,ik->jk', train_x, train_x)
        hist_dict['elbo'] = torch.zeros(hist_length, dtype=self.dtype)

        #print("Start calculation initial ELBO value")
        self.elbo = self._get_elbo(train_x)
        hist_dict['elbo'][0] = self.elbo
        self.optimizer = optim.RMSprop(self.torch_vars, 
                                       lr=self.params['rms_eta'])

        # Now move to training loop
        #print("Start training loop")
        #print(self.torch_vars)
        t0 = time.time()

        for i in range(self.params['n_iter']):
            self.elbo = self._get_elbo(train_x)
            loss = -self.elbo
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            #print("Made optimization step!")
            #print(self.torch_vars)

            # Record the information more often than we print
            if (i+1) % self.params['save_every'] == 0:
                save_idx = int((i+1) / self.params['save_every'])

                # Record the updated data in the history
                for j in range(len(self.torch_vars)):
                    value = self.torch_vars[j].clone().detach().cpu()
                    if len(value.shape) > 0:
                        value = value.numpy()
                        hist_dict[self.var_names[j]][save_idx, :] = value
                    else:
                        value = value.item()
                        hist_dict[self.var_names[j]][save_idx] = value


                hist_dict['elbo'] = self.elbo.detach().cpu().item()

                # Assume params['print_every'] divides params['save_every']
                if (i+1) % self.params['print_every'] == 0:
                    print(('{0}, d: {1:d}, Iter: {2:d}-{3:d}, s/iter:'
                        + ' {4:.3e}, ELBO: {5:.3e}').format(
                        self.model_name,
                        self.d,
                        train_ind+1,
                        i+1,
                        (time.time()-t0) / self.params['print_every'],
                        hist_dict['elbo']
                        )
                    )
                    t0 = time.time()

        # Save the data
        save_file = os.path.join(self.save_dir, 
            '{0}_train_{1:d}.p'.format(self.model_name, train_ind))
        pickle.dump(hist_dict, open(save_file, 'wb'))

        delta = self.torch_vars[0].clone().detach().cpu().numpy()
        log_sigma = self.torch_vars[1].clone().detach().cpu().numpy()
        sigma = np.exp(log_sigma)

        return (delta, sigma)

    def _get_yk_z_sig(self, x, z_in):
        """ 
        Method needed to calculate NLLs common to all models 
        We want to calculate, for each z in the batch, the sum of
            (x_k - mu_X - z)^T * Sigma_X^{-1} * (x_k - mu_X - z)
        over k = 1, ..., N, where:
            - Sigma_X is the estimated model covariance matrix
            - mu_X is the estimated model offset
            - N is the number of datapoints
        Returns:
            yk_z_sig: Tensorflow vector of length self.tf_batch
        """
        yk_z_sig = torch.zeros(self.n_batch)

        for k in range(self.params['n_data']):
            x_k = x[k ,:]
            yk_z_sig += torch.sum(
                (x_k-self.delta-z_in)**2 * torch.exp(-2*self.log_sigma), 1)

        return yk_z_sig

    def _get_elbo(self, x):
        raise NotImplementedError
