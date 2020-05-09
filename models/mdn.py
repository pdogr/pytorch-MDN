import torch
import torch.nn as nn
import torch.nn.functional as F


class MixtureDensityNetwork(nn.Module):
	"""
    Mixture density network.

    [ Bishop, 1994 ]
    Parameters
    ----------
    dim_in; dimensionality of the input 
    dim_out: int; dimensionality of the output
    num_latent: int; number of components in the mixture model

    Output
    ----------
    (pi,mu,sigma) 
    pi (batch_size x num_latent) is prior 
    mu (batch_size x dim_out x num_latent) is mean of each Gaussian

    sigma (batch_size x dim_out x num_latent) is standard deviation of each Gaussian

    """	
    def __init__(self,dim_in,dim_out,num_latent):
		super(MixtureDensityNetwork,self).__init__()
		self.dim_in=dim_in
		self.hidden_dim= hidden_dim
		self.num_latent=num_latent
		self.pi_h=nn.Linear(dim_in,num_latent),
		self.mu_h=nn.Linear(dim_in,dim_out*num_latent)
		self.sigma_h=nn.Linear(dim_in,dim_out*num_latent)

	@staticmethod
	def forward(self,x):

		pi=F.softmax(self.pi_h(x))

		mu=self.mu_h(x)
		mu=mu.view(-1,self.dim_out,self.num_latent)

		sigma=torch.exp(self.sigma_h(x))
		sigma=sigma.view(-1,self.dim_out,self.num_latent)


		return pi,mu,sigma












