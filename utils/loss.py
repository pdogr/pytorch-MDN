import torch
import torch.nn.functional as F
from utils import gaussian_distribution

def MDN_loss(y,pi,mu,sigma):
	g=gaussian_distribution(y,mu,sigma)
	assert(g.shape==p.shape)
	prob=pi*g
	prob=torch.prod(prob,1)
	nll=-torch.log(torch.sum(prob,dim=1))
	return torch.mean(nll)



