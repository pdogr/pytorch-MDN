import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.mdn import MixtureDensityNetwork
from utils.utils import sample
from utils.loss import MDN_loss

import seaborn as sns; sns.set()
import matplotlib.pyplot as plt




"""
Example from paper 
	x=t+0.3sin(2*pi*t)+e
"""
def generate_data(num_samples=1000):
	t=np.random.uniform(0,1,num_samples)
	e=np.random.uniform(-0.1,0.1,num_samples)
	x=t+0.3*np.sin(2*np.pi*t)+e
	return t,x



if __name__=='__main__':

	num_samples=1000
	t,x=generate_data(num_samples)
	plt.scatter(t,x,alpha=0.2,label='Original')
	plt.savefig('img/original.jpg')
	plt.xlabel('t')
	plt.ylabel('x')
	plt.legend()
	plt.show()



	model=nn.Sequential(
		nn.Linear(1,20),
		nn.Tanh(),
		MixtureDensityNetwork(20,1,5),
		)
	t=t.reshape((num_samples,1)).astype(np.float32)
	x=x.reshape((num_samples,1)).astype(np.float32)

	x_var=torch.from_numpy(x)
	t_var=torch.from_numpy(t)

	plt.scatter(x,t,alpha=0.2,label='Inverse')
	plt.savefig('img/inverse.jpg')
	plt.xlabel('x')
	plt.ylabel('t')
	plt.legend()
	plt.show()

	opt=optim.Adam(model.parameters(),lr=0.012)
	for e in range(7000):
		opt.zero_grad()
		pi,mu,sigma=model.forward(x_var)
		loss=MDN_loss(t_var,pi,mu,sigma)
		loss.backward()
		opt.step()
		if e%500==0:
			print('Epoch: {0}\t Loss: {1}'.format(e,loss.item()))

	num_test=1000
	x_test=np.linspace(-0.2,1.2,num_test).astype(np.float32)
	x_test=x_test.reshape(num_test,1)
	x_test=torch.from_numpy(x_test)

	pi,mu,sigma=model.forward(x_test)

	samples=sample(pi,mu,sigma)
	plt.scatter(x,t,alpha=0.2,color='b',label='Inverse')

	plt.scatter(x_test,samples,alpha=0.2,color='r',label='Sampled')
	plt.xlabel('x')
	plt.ylabel('t')
	plt.legend()

	plt.savefig('img/orig+sampled.jpg')
	plt.show()




















