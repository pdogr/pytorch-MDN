import torch
import numpy as np
import torch.nn as nn
import math
from torch.distributions import Categorical
from torch.autograd import Variable
def gaussian_distribution(y,mu,sigma):

	"""
	# Parameters
    ----------
    y (batch_size x dim_out): vector of responses
    mu (batch_size x dim_out x num_latent) is mean of each Gaussian
    sigma (batch_size x dim_out x num_latent) is standard deviation of each Gaussian

    Output
    ----------
	Gaussian distribution (batch_size x dim_out x num_latent)

	"""	
	y=y.unsqueeze(2).expand_as(mu)
	one_div_sqrt_pi=1.0/math.sqrt(2.0*np.pi)

	x=(y.expand_as(mu)-mu)*torch.reciprocal(sigma)
	x=torch.exp(-0.5*x*x)*one_div_sqrt_pi
	x=x*torch.reciprocal(sigma)
	return torch.prod(x,1)


def sample(pi,mu,sigma):

	"""
	# Parameters
    ----------
    pi (batch_size x num_latent) is priors
    mu (batch_size x dim_out x num_latent) is mean of each Gaussian
    sigma (batch_size x dim_out x num_latent) is standard deviation of each Gaussian

    Output
    ----------

	"""	
	cat=Categorical(pi)
	ids=list(cat.sample().data)
	sampled=Variable(sigma.data.new(sigma.size(0),
					sigma.size(1)).normal_())
	for i,idx in enumerate(ids):
		sampled[i]=sampled[i].mul(sigma[i,:,idx]).add(mu[i,:,idx])
	return sampled.data.numpy()




