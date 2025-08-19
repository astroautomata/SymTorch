
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt 

from symtorch import MLP_SR

from pinn_script import RegularNN, PINN

pinn = PINN()
pinn.load_state_dict(torch.load('pinn.pth'))


regular_NN = RegularNN()
regular_NN.load_state_dict(torch.load('regular_NN.pth'))


pinn = pinn.to(torch.device('mps'))
regular_NN = regular_NN.to(torch.device('mps'))

num_data = 5000
sample_data = torch.tensor(np.stack([np.linspace(0,1,num_data), np.linspace(0,1,num_data)], axis=1), dtype=torch.float32) 


regular_NN.net = MLP_SR(regular_NN.net, 'nn')
pinn.net = MLP_SR(pinn.net, 'pinn')

sr_params = {'niterations': 1000,
             'constraints': {'sin':3, 'exp':3}, 
             'complexity_of_operators': {'sin':3, 'exp':3},
             "unary_operators": ["inv(x) = 1/x", "sin", "exp"],
             'parsimony': 0.01,
             'nested_constraints':{'sin':{'sin':0, 'exp':0}, 'exp':{'exp':0, 'sin':0}},
            #  'complexity_of_constants':2
             }

variable_names = ['x', 't']
fit_params = {'variable_names': variable_names}

pinn.net.distill(sample_data.to(torch.device('mps')), sr_params = sr_params,
                 fit_params=fit_params
                 )

regular_NN.net.distill(sample_data.to(torch.device('mps')), sr_params = sr_params,
                 fit_params=fit_params
                 )
