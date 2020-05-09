import torch
import numpy as np
import torch.nn as nn
import math


def gaussian_distribution(y,mu,sigma):


# 	"""
# 	# Parameters
#  #    ----------
#  #    y (batch_size x dim_out): vector of responses
#  #    mu (batch_size x dim_out x num_latent) is mean of each Gaussian
#  #    sigma (batch_size x dim_out x num_latent) is standard deviation of each Gaussian

#  #    Output
#  #    ----------
# 	Gaussian distribution (batch_size x dim_out x num_latent)

# """
	y=y.unsqueeze(2).expand_as(mu)
	one_div_sqrt_pi=1.0/math.sqrt(2*np.pi)

	x=(y.expand_as(mu)-mu)*torch.reciprocal(sigma)
	x=torch.exp(-0.5*x*x)*one_div_sqrt_pi
	return x