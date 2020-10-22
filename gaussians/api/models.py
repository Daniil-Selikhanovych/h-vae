import numpy as np

import torch
from torch.distributions.normal import Normal
import torch.optim as optim
from torch.nn import Parameter
from torch import nn

import time
import pickle
import os

class BaseModel(nn.Module):
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
        super().__init__()
        # Variable initialization
        self.dtype = var_inits[0].dtype
        self.device = var_inits[0].device
        
        torch_var_inits = []
        torch_vars = []
        for i in range(len(var_inits)):
            cur_params = var_inits[i].clone().detach().to(self.device)
            torch_var_inits.append(cur_params)
            torch_vars.append(Parameter(cur_params))

        self.params = params
        self.var_names = var_names
        self.torch_vars = torch_vars
        self.var_inits = torch_var_inits
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
            
    def train(self, train_x, train_ind, 
              t_delta=None, t_sigma=None,
              clip_value=None):
        """ 
        Train the model.
        Args:
            train_x: PyTorch training data
            train_ind: Index of training run
        Returns:
            delta: Value of delta at end of training
            sigma: Value of sigma at end of training
        """      

        # Start by initializing parameter and ELBO history
        hist_length = int(self.params['n_iter']/self.params['save_every']) + 1
        hist_dict = {}

        hist_dict['elbo'] = torch.zeros(hist_length, dtype=self.dtype).numpy()
        if (t_delta is not None):
            hist_dict['diff_delta'] = torch.zeros(hist_length, 
                                                  dtype=self.dtype).numpy()
            delta = self.torch_vars[0].clone().detach().cpu()
            delta_diff = torch.sum((delta - t_delta.detach().cpu())**2).item()
            hist_dict['diff_delta'][0] = delta_diff
            print(f"Init diff delta = {delta_diff}")

        if (t_sigma is not None):
            hist_dict['diff_sigma'] = torch.zeros(hist_length, 
                                                  dtype=self.dtype).numpy()
            sigma = self.torch_vars[1].clone().detach().cpu().exp()
            sigma_diff = torch.sum((sigma - t_sigma.detach().cpu())**2).item()
            hist_dict['diff_sigma'][0] = sigma_diff  
            print(f"Init diff sigma = {sigma_diff}")       

        if (t_delta is not None) and (t_sigma is not None):
            hist_dict['diff_theta'] = torch.zeros(hist_length, 
                                                  dtype=self.dtype).numpy()
            theta_diff = delta_diff + sigma_diff                                    
            hist_dict['diff_theta'][0] = theta_diff                                      
            print(f"Init diff theta = {theta_diff}")               

        #print("Start calculation initial ELBO value")
        elbo = self._get_elbo(train_x).detach().cpu().item()
        hist_dict['elbo'][0] = elbo
        optimizer = optim.RMSprop(self.torch_vars, lr=self.params['rms_eta'])

        # Now move to training loop
        #print("Start training loop")
        #print(self.torch_vars)
        t0 = time.time()

        for i in range(self.params['n_iter']):
            #train_x = train_x.detach()
            if (i+1) % self.params['print_every'] == 0:
                elbo = self._get_elbo(train_x, print_results=True)
            else:
                elbo = self._get_elbo(train_x, print_results=False)
            loss = -elbo
            optimizer.zero_grad()
            loss.backward()
            if clip_value is not None:
                torch.nn.utils.clip_grad_norm_(self.parameters(), clip_value)
            optimizer.step()
            #print("Made optimization step!")
            #print(self.torch_vars)

            # Record the information more often than we print
            if (i+1) % self.params['save_every'] == 0:
                save_idx = (i+1) // self.params['save_every']
                #print("Save results...")
                if (t_delta is not None):
                    delta = self.torch_vars[0].clone().detach().cpu()
                    delta_diff = torch.sum((delta - \
                                            t_delta.detach().cpu())**2).item()
                    hist_dict['diff_delta'][save_idx] = delta_diff

                if (t_sigma is not None):
                    sigma = self.torch_vars[1].clone().detach().cpu().exp()
                    sigma_diff = torch.sum((sigma - \
                                            t_sigma.detach().cpu())**2).item()
                    hist_dict['diff_sigma'][save_idx] = sigma_diff 

                if (t_delta is not None) and (t_sigma is not None):
                    theta_diff = delta_diff + sigma_diff                                    
                    hist_dict['diff_theta'][save_idx] = theta_diff   
                
                hist_dict['elbo'][save_idx] = elbo.detach().cpu().item()

            # Assume params['print_every'] divides params['save_every']
            if (i+1) % self.params['print_every'] == 0:
                print(('{0}, d: {1:d}, Iter: {2:d}-{3:d}, s/iter:'
                    + ' {4:.3e}, ELBO: {5:.3e}').format(
                    self.model_name,
                    self.d,
                    train_ind+1,
                    i+1,
                    (time.time()-t0) / self.params['print_every'],
                    elbo.clone().detach().cpu().item()
                    )
                )
                t0 = time.time()
                if (t_delta is not None):
                    delta = self.torch_vars[0].clone().detach().cpu()
                    delta_diff = torch.sum((delta - \
                                            t_delta.detach().cpu())**2).item()
                    print(f"Cur diff delta = {delta_diff}")

                if (t_sigma is not None):
                    sigma = self.torch_vars[1].clone().detach().cpu().exp()
                    sigma_diff = torch.sum((sigma - \
                                            t_sigma.detach().cpu())**2).item()
                    print(f"Cur diff sigma = {sigma_diff}")
                
                if (t_delta is not None) and (t_sigma is not None):
                    theta_diff = delta_diff + sigma_diff
                    print(f"Cur diff theta = {theta_diff}") 
                print("---------------------------------------")

        # Save the data
        save_file = os.path.join(self.save_dir, 
            '{0}_train_{1:d}.p'.format(self.model_name, train_ind))
        pickle.dump(hist_dict, open(save_file, 'wb'))

        delta = self.torch_vars[0].clone().detach().cpu().numpy()
        log_sigma = self.torch_vars[1].clone().detach().cpu().numpy()
        sigma = np.exp(log_sigma)

        return (delta, sigma)

    def _get_yk_z_sig(self, x, z_in):
        yk_z_sig = torch.zeros(self.n_batch)

        for k in range(self.params['n_data']):
            x_k = x[k ,:]
            yk_z_sig += torch.sum(
                (x_k-self.delta-z_in)**2 * torch.exp(-2*self.log_sigma), 1)

        return yk_z_sig

    def _get_elbo(self, x, print_results=False):
        raise NotImplementedError
        
    def _reinitialize(self, params):
        raise NotImplementedError
        
class HVAE(BaseModel):
    """ Specific implementation of HVAE """

    def __init__(self, params, var_names, var_inits, model_name, d, K):
        """ Initialize model including ELBO calculation """

        super().__init__(params, var_names, var_inits, model_name, d)

        self.K = K
        self.logit_eps = self.torch_vars[2]

        # If there are only three variables, it means we are not tempering
        if len(var_names) == len(var_inits) == 3:
            self.tempering = False
        else:
            self.tempering = True
            self.log_T_0 = self.torch_vars[-1]


    def _his(self, x_bar):
        """ 
        Perform the HIS step to evolve samples
        Returns:
            z_0: Initial position
            p_0: Initial momentum
            z_K: Final position
            p_K: Final momentum
        """
        z_graph = {}
        p_graph = {}

        x_bar = x_bar.clone().detach().to(self.device)

        # Sample initial values with reparametrization if necessary
        z_0 = self.std_norm.sample([self.n_batch]).to(self.device)
        gamma_0 = self.std_norm.sample([self.n_batch]).to(self.device)

        if not self.tempering:
            p_0 = gamma_0
        else:
            p_0 = (1 + torch.exp(self.log_T_0))*gamma_0

        z_graph[0] = z_0
        p_graph[0] = p_0

        # Initialize temperature, define step size
        if not self.tempering:
            T_km1 = 1. # Keep the temperature at 1 throughout if not tempering
            T_k = 1.
        else:
            T_km1 = 1 + torch.exp(self.log_T_0)

        epsilon = self.params['max_eps'] / (1 + torch.exp(-self.logit_eps))
        var_x = torch.exp(2 * self.log_sigma)

        # Now perform K alternating steps of leapf.to(self.device)rog and cooling
        for k in range(1, self.K+1):
                
            # First perform a leapfrog step
            z_in = z_graph[k-1]
            p_in = p_graph[k-1]

            p_half = p_in - 1/2*epsilon*self._dU_dz(z_in, x_bar, var_x)
            z_k = z_in + epsilon*p_half
            p_temp = p_half - 1/2*epsilon*self._dU_dz(z_k, x_bar, var_x)

            # Then do tempering
            if self.tempering: 
                T_k = 1 + torch.exp(self.log_T_0)*(1 - k**2/self.K**2)

            p_k = T_k/T_km1 * p_temp

            # End with updating the graph and the previous temperature
            z_graph[k] = z_k
            p_graph[k] = p_k
            T_km1 = T_k

        # Extract final (z_K, p_K)
        z_K = z_graph[self.K]
        p_K = p_graph[self.K]

        return (z_0, p_0, z_K, p_K)

    def _dU_dz(self, z_in, x_bar, var_x):
        """ Calculate the gradient of the potential wrt z_in """
        grad_U = (z_in 
            + self.params['n_data']*(z_in + self.delta - x_bar)/var_x)
        return grad_U

    def _get_elbo(self, x, print_results=False):
        """ 
        Calculate the ELBO for HVAE 
        Args:
            x: data
        Returns:
            elbo: The ELBO objective as a PyTorch object
        """
        x_bar = torch.mean(x, 0)
        C_xx = torch.einsum('ij,ik->jk', x, x)

        (z_0, p_0, z_K, p_K) = self._his(x_bar)

        var_inv_vec = torch.exp(-2*self.log_sigma)
        var_inv_mat = torch.diag(var_inv_vec)
        trace_term = torch.trace(torch.matmul(var_inv_mat, C_xx))

        z_sigX_z = torch.sum((z_K + self.delta) * var_inv_vec * 
            (z_K + self.delta - 2*x_bar), 1)
        z_T_z = torch.sum(z_K*z_K, 1)
        p_T_p = torch.sum(p_K*p_K, 1)

        Nd2_log2pi = self.params['n_data']*self.d/2*np.log(2*np.pi)

        elbo = (-self.params['n_data']*torch.sum(self.log_sigma) 
            - trace_term/2 - self.params['n_data']/2*torch.mean(z_sigX_z)
            - 1/2*(torch.mean(z_T_z) + torch.mean(p_T_p))
            + self.d - Nd2_log2pi)

        return elbo
    
    def _reinitialize(self, params):
        torch_var_inits = []
        torch_vars = []
        for i in range(len(params)):
            cur_params = params[i].clone().detach().to(self.device)
            torch_var_inits.append(cur_params)
            torch_vars.append(Parameter(cur_params))
            
        self.torch_vars = torch_vars
        self.var_inits = torch_var_inits
        self.delta = torch_vars[0]
        self.log_sigma = torch_vars[1]
        self.logit_eps = self.torch_vars[2]

        # If there are only three variables, it means we are not tempering
        if len(self.var_names) == len(params) == 3:
            self.tempering = False
        else:
            self.tempering = True
            self.log_T_0 = self.torch_vars[-1]
        
        
class NF(BaseModel):
    """ Specific implementation of the Normalizing Flow """

    def __init__(self, params, var_names, var_inits, model_name, d, K):
        """ Initialize model including ELBO calculation """

        super().__init__(params, var_names, var_inits, model_name, d)

        self.K = K
        self.u_pre_reparam = self.torch_vars[2]
        self.w = self.torch_vars[3]
        self.b = self.torch_vars[4]


    def _nf(self, u):
        """ 
        Perform the flow step to get the final samples
        Returns:RMSprop
            z_0: Initial sample from variational prior
            z_K: Final sample after normalizing flow
            log_det_sum: Sum of log determinant terms at each flow step
        """
        z_graph = {}
        log_det_sum = torch.tensor(0., dtype=self.dtype).to(self.device)

        # Begin with sampling from the variational prior
        z_0 = self.std_norm.sample([self.n_batch]).to(self.device)
        z_graph[0] = z_0

        # Need awkward operations to deal with broadcasting
        u_tiled = torch.reshape(u, (self.d, 1)).repeat([1, self.n_batch])

        # Now perform the flow steps
        for i in range(1, self.K+1):

            # Evolution bit
            z_in = z_graph[i-1]

            w_T_z = torch.sum(self.w * z_in, 1)
            post_tanh = torch.tanh(w_T_z + self.b)
            u_tanh = torch.transpose(u_tiled * post_tanh, 0, 1)

            z_out = z_in + u_tanh
            z_graph[i] = z_out

            # log determinant terms 
            u_T_w = torch.sum(u * self.w)
            log_det_batch = torch.log(1. + (1 - post_tanh**2) * u_T_w)

            log_det_sum += torch.mean(log_det_batch)

        # Extract final z_K
        z_K = z_graph[self.K]

        return (z_0, z_K, log_det_sum)

    def _get_elbo(self, x, print_results=False):
        """ 
        Calculate the ELBO for NF 
        Args:
            x: data
        Returns:
            elbo: The ELBO objective as a tensorflow object
        """
        w_T_u_pre_reparam = torch.sum(self.u_pre_reparam * self.w)
        u = (self.u_pre_reparam + \
                            (-1 + nn.Softplus()(w_T_u_pre_reparam) 
            - w_T_u_pre_reparam)*self.w / torch.sum(self.w**2))

        (z_0, z_K, log_det_sum) = self._nf(u)
        var_inv_vec = torch.exp(-2 * self.log_sigma)

        # Note that we say y = x - mu_X for ease of naming
        #x = x.clone().detach().to(self.device)
        x_bar = torch.mean(x, 0)

        y_sig_y = torch.sum((x - self.delta)**2 * var_inv_vec)
        y_bar_sig_z = torch.sum((x_bar - self.delta) 
                                    * var_inv_vec * z_K, 1)
        mean_y_bar_sig_z = torch.mean(y_bar_sig_z)
        z_sig_z = torch.sum(z_K**2 * var_inv_vec, 1)
        mean_z_sig_z = torch.mean(z_sig_z) 
        z_T_z = torch.sum(z_K * z_K, 1)
        mean_z_T_z = torch.mean(z_T_z)

        Nd2_log2pi = self.d/2*np.log(2*np.pi)
        
        elbo = (- Nd2_log2pi - torch.sum(self.log_sigma)
            - y_sig_y/self.params['n_data'] + mean_y_bar_sig_z
            - 1./2.*mean_z_sig_z 
            - (1./2*mean_z_T_z)/self.params['n_data'] 
            + log_det_sum/self.params['n_data'])*self.params['n_data']
        
        if print_results:
            print_log_det_sum = log_det_sum.clone().detach().cpu().item()
            print_y_sig_y = y_sig_y.clone().detach().cpu().item()
            print_mean_z_sig_z = mean_z_sig_z.clone().detach().cpu().item()
            print_mean_z_T_z = mean_z_T_z.clone().detach().cpu().item()
            print_mean_y_bar_sig_z = mean_y_bar_sig_z.clone().detach().cpu().item()
            print(f"log_det_sum = {print_log_det_sum}, y_sig_y = {print_y_sig_y}")
            print(f"mean_z_sig_z = {print_mean_z_sig_z}, mean_z_T_z = {print_mean_z_T_z}")
            print(f"mean_y_bar_sig_z = {print_mean_y_bar_sig_z}")

        return elbo
        
    def _reinitialize(self, params):
        torch_var_inits = []
        torch_vars = []
        for i in range(len(params)):
            cur_params = params[i].clone().detach().to(self.device)
            torch_var_inits.append(cur_params)
            torch_vars.append(Parameter(cur_params))
            
        self.torch_vars = torch_vars
        self.var_inits = torch_var_inits
        self.delta = torch_vars[0]
        self.log_sigma = torch_vars[1]
        self.u_pre_reparam = self.torch_vars[2]
        self.w = self.torch_vars[3]
        self.b = self.torch_vars[4]
        
class VB(BaseModel):
    """ Specific implementation of Variational Bayes """

    def __init__(self, params, var_names, var_inits, model_name, d):
        """ Initialize model including ELBO calculation """

        super().__init__(params, var_names, var_inits, model_name, d)

        self.mu_z = self.torch_vars[2]
        self.log_sigma_z = self.torch_vars[3]

    def _get_elbo(self, x, print_results=False):
        """ 
        Calculate the ELBO for the VB example
        Returns:
            elbo: The ELBO objective as a tensorflow object
        """
        var_inv_vec = torch.exp(-2 * self.log_sigma)

        # Note that for this VB scheme the ELBO is completely deterministic
        y_sig_y = torch.sum((x - self.delta)**2 * var_inv_vec)
        y_sig_mu = torch.sum((x - self.delta) * var_inv_vec * self.mu_z)

        var_Z_over_var_X = torch.sum(torch.exp(2*self.log_sigma_z)* var_inv_vec)

        mu_sig_mu = torch.sum(self.mu_z**2 * var_inv_vec)
        mu_T_mu = torch.sum(self.mu_z**2)

        Nd2_log2pi = params['n_data']*self.d/2*np.log(2*np.pi)

        elbo = (- Nd2_log2pi + torch.sum(self.log_sigma_z) 
            - params['n_data']*torch.sum(self.log_sigma) - 1/2*y_sig_y 
            + y_sig_mu - params['n_data']/2*(var_Z_over_var_X + mu_sig_mu)
            - 1/2*torch.sum(torch.exp(2*self.log_sigma_z)) - 1/2*mu_T_mu 
            - self.d/2
            )

        return elbo
        
    def _reinitialize(self, params):
        torch_var_inits = []
        torch_vars = []
        for i in range(len(params)):
            cur_params = params[i].clone().detach().to(self.device)
            torch_var_inits.append(cur_params)
            torch_vars.append(Parameter(cur_params))
            
        self.torch_vars = torch_vars
        self.var_inits = torch_var_inits
        self.delta = torch_vars[0]
        self.log_sigma = torch_vars[1]
        self.mu_z = self.torch_vars[2]
        self.log_sigma_z = self.torch_vars[3]
