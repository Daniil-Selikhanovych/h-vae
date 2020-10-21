import torch
from base_model import BaseModel

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


    def _his(self):
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

        # Sample initial values with reparametrization if necessary
        z_0 = self.std_norm.sample([self.n_batch])
        gamma_0 = self.std_norm.sample([self.n_batch])

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

        # Now perform K alternating steps of leapfrog and cooling
        for k in range(1, self.K+1):
                
            # First perform a leapfrog step
            z_in = z_graph[k-1]
            p_in = p_graph[k-1]

            p_half = p_in - 1/2*epsilon*self._dU_dz(z_in, var_x)
            z_k = z_in + epsilon*p_half
            p_temp = p_half - 1/2*epsilon*self._dU_dz(z_k, var_x)

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

    def _dU_dz(self, z_in, var_x):
        """ Calculate the gradient of the potential wrt z_in """
        grad_U = (z_in 
            + self.params['n_data']*(z_in + self.delta - self.x_bar)/var_x)
        return grad_U

    def _get_elbo(self, x):
        """ 
        Calculate the ELBO for HVAE 
        Args:
            z_K: Final position after HIS evolution
            p_K: Final momentum after HIS evolution
        Returns:
            elbo: The ELBO objective as a tensorflow object
        """
        x = torch.tensor(x, dtype=torch.float32)
        self.x_bar = torch.sum(x, 0)
        self.C_xx = torch.einsum('ij,ik->jk', x, x)

        (z_0, p_0, z_K, p_K) = self._his()

        var_inv_vec = torch.exp(-2*self.log_sigma)
        var_inv_mat = torch.diag(var_inv_vec)
        trace_term = torch.trace(torch.matmul(var_inv_mat, self.C_xx))

        z_sigX_z = torch.sum((z_K + self.delta) * var_inv_vec * 
            (z_K + self.delta - 2*self.x_bar), 1)
        z_T_z = torch.sum(z_K*z_K, 1)
        p_T_p = torch.sum(p_K*p_K, 1)

        Nd2_log2pi = self.params['n_data']*self.d/2*np.log(2*np.pi)

        elbo = (-self.params['n_data']*torch.sum(self.log_sigma) 
            - trace_term/2 - self.params['n_data']/2*torch.mean(z_sigX_z)
            - 1/2*(torch.mean(z_T_z) + torch.mean(p_T_p))
            + self.d - Nd2_log2pi)

        return elbo
