import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from torch.optim import lr_scheduler
import numpy as np
import time
import matplotlib
from matplotlib import pyplot as pp
import scipy
from scipy.integrate import solve_ivp
import random
import functools
#from training_data import ground_truth

### FCN for forward PINNs

class FCNforward(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3, activation_fn=nn.Tanh):
        super(FCNforward, self).__init__()
        
        self.activation_fn = activation_fn() # Instantiate the activation function
        
        # Input layer definition
        self.inputlayer = nn.Linear(input_size, hidden_size)
        
        # Hidden layers definition using nn.ModuleList
        self.hiddenlayers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers - 1)
        ])
        
        # Output layer definition
        self.outputlayer = nn.Linear(hidden_size, output_size)
        
        # Initialize weights and biases
        # Combine all layers into one list for easy iteration
        all_layers = [self.inputlayer, *self.hiddenlayers, self.outputlayer]
        
        for layer in all_layers:
            # Weights from a normal distribution (Xavier normal)
            # Recommended gain value for tanh is used by default if activation is specified
            nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain('tanh'))
            # Set biases to zero
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        # Pass input through the input layer and activation
        x = self.activation_fn(self.inputlayer(x))
        
        # Pass through hidden layers with activation
        for layer in self.hiddenlayers:
            x = self.activation_fn(layer(x))
            
        # Pass through the final output layer (no activation here, often preferred for regression/discovery)
        x = self.outputlayer(x)
        
        return x

class FCN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3, activation=nn.Tanh):
        super(FCN, self).__init__()
        self.inputlayer = nn.Sequential(nn.Linear(input_size, hidden_size),activation())
        self.hiddenlayers = nn.Sequential(*[nn.Sequential(*[nn.Linear(hidden_size, hidden_size),activation()]) for _ in range(num_layers - 1)])
        self.outputlayer = nn.Linear(hidden_size, output_size)
                                        
    def forward(self, x):
        x = self.inputlayer(x)
        x = self.hiddenlayers(x)
        x = self.outputlayer(x)
        return x




### Solution only with two learnable parameter

class FCNinverse(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3, activation_fn=nn.Tanh):
        super(FCNinverse, self).__init__()
        
        self.activation_fn = activation_fn() # Instantiate the activation function
        
        # Input layer definition
        self.inputlayer = nn.Linear(input_size, hidden_size)
        
        # Hidden layers definition using nn.ModuleList
        self.hiddenlayers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers - 1)
        ])
        
        # Output layer definition
        self.outputlayer = nn.Linear(hidden_size, output_size)
        
        # Initialize weights and biases
        # Combine all layers into one list for easy iteration
        all_layers = [self.inputlayer, *self.hiddenlayers, self.outputlayer]
        
        for layer in all_layers:
            # Weights from a normal distribution (Xavier normal)
            # Recommended gain value for tanh is used by default if activation is specified
            nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain('tanh'))
            # Set biases to zero
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        # Define learnable parameters mu and omega
        # Use nn.Parameter for tensors that should be part of the model's parameters
        self.mu = nn.Parameter(data=torch.tensor([0.01]))
        self.omega = nn.Parameter(data=torch.tensor([0.01]))

    def forward(self, x):
        # Pass input through the input layer and activation
        x = self.activation_fn(self.inputlayer(x))
        
        # Pass through hidden layers with activation
        for layer in self.hiddenlayers:
            x = self.activation_fn(layer(x))
            
        # Pass through the final output layer (no activation here, often preferred for regression/discovery)
        x = self.outputlayer(x)
        
        return x
    

### Autograd to produce derivate. 
def derivates(y,t,order):
    """ This function computes the derivatives of y with respect to x using PyTorch's autograd.
    Args:   y (torch.Tensor): The output tensor for which derivatives are computed. In our case it will be 1. 
            t (torch.Tensor): The input tensor with respect to which derivatives are computed. in this case it is the time t.
            order (int): The order of the derivative to compute. Default is 1.
    Returns:
        torch.Tensor: The computed derivative of y with respect to x.
    Note: y actually has to depend on x for the autograd to work properly.Otherwise it throws an error. 
    """
    for _ in range(order):
        y = torch.autograd.grad(y, t, grad_outputs=torch.ones_like(y,device=y.device),
                                create_graph=True,retain_graph=True)[0] ## needed if called multiple times. 
    return y


  
### ic loss
def ic_loss(model,tboundary, y0,dy_dt0):
    """This function computes the initial condition loss for a given model and input.
    Args:
        model (nn.Module): The neural network model.
        t (torch.Tensor): Input tensor.
        y0 (torch.Tensor): Initial condition tensor.
    Returns:
        torch.Tensor: The computed initial condition loss."""
    y = model(tboundary)  # Get model output at t=0
    lossic1 = torch.mean((y - y0)**2)  
    dydt = derivates(y, tboundary, 1)
    lossic2 = torch.mean((dydt - dy_dt0)**2) # Initial condition for y at t=0
   # Get the model output at t=0
    return lossic1,lossic2  # Mean Squared Error for initial condition



###################Forward model #############################################################
### physics loss
def physics_loss(model, t_phys, delta, omega0):
    """This function computes the physics loss for a given model and input.
    Args:
        model (nn.Module): The neural network model.
        x (torch.Tensor): Input tensor.
        delta (float): Damping factor.
        omega0 (float): Natural frequency.
    Returns:
        torch.Tensor: The computed physics loss."""
    y = model(t_phys)
    ydot = derivates(y, t_phys, 1)
    yddot = derivates(y, t_phys,2)
    # Physics equation: y'' + 2*delta*y' + omega0^2*y = 0
    residual = yddot + 2 * delta * ydot + omega0**2 * y
    return torch.mean(residual**2)


### physics loss now with timeless residual to overcome stiff PINNs
def physics_loss_scaled(model, t_phys, delta, omega0):
    """This function computes the physics loss for a given model and input.
    Args:
        model (nn.Module): The neural network model.
        x (torch.Tensor): Input tensor.
        delta (float): Damping factor.
        omega0 (float): Natural frequency.
    Returns:
        torch.Tensor: The computed physics loss."""
    zeta = delta/omega0
    y = model(t_phys)
    ydot = derivates(y, t_phys, 1)
    yddot = derivates(y, t_phys,2)
    # Physics equation: y'' + 2*delta*y' + omega0^2*y = 0
    residual = yddot + 2 * zeta * omega0 * ydot + omega0**2 * y
    return torch.mean(residual**2)


### physics loss averaged
def physics_loss_averaged(model, t_phys, delta, omega0):

    y = model(t_phys)

    ydot = derivates(y, t_phys, 1)
    yddot = derivates(y, t_phys,2)
    
    # Normalize residual
    residual = (
        yddot
        + 2*delta*ydot
        + omega0**2*y
    )

    scale = (
        torch.mean(yddot**2)
        + torch.mean((2*delta*ydot)**2)
        + torch.mean((omega0**2*y)**2)
        + 1e-8 # add a tiny floor to alow division
    )

    return torch.mean(residual**2) / scale


###################Inverse model #############################################################

### physics loss for inverse problem with learnable parameters
def physics_loss_inverse(model, t_phys):
    """This function computes the physics loss for a given model and input.
    Args:
        model (nn.Module): The neural network model.
        x (torch.Tensor): Input tensor.
        delta (float): Damping factor.
        omega0 (float): Natural frequency.
    Returns:
        torch.Tensor: The computed physics loss."""
    y = model(t_phys)
    ydot = derivates(y, t_phys, 1)
    yddot = derivates(y, t_phys,2)
    # Physics equation: y'' + 2*delta*y' + omega0^2*y = 0
    residual = yddot + 2 * model.mu * ydot + model.omega**2 * y # adding the learnable parameter r to learn omega
    return torch.mean(residual**2)

if __name__ == "__main__":
    pass # for testing purposes only
    model = FCNforward(1,32,1)
    print(model)