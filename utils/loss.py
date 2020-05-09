import torch
import torch.nn.functional as F
from utils import gaussian_distribution
from torch.distributions import Categorical

def MDN_loss(y,pi,mu,sigma):

	"""
	# Parameters
    ----------
    y (batch_size x dim_out): vector of responses 
    mu (batch_size x dim_out x num_latent) is priors
    mu (batch_size x dim_out x num_latent) is mean of each Gaussian
    sigma (batch_size x dim_out x num_latent) is standard deviation of each Gaussian

    Output
    ----------
    negative log likelihood loss

	"""	

	g=gaussian_distribution(y,mu,sigma)
	prob=pi*g
	prob=torch.prod(prob,1)
	nll=-torch.log(torch.sum(prob,dim=-1))
	return torch.mean(nll)



