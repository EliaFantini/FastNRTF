import torch
import torch.nn as nn
import numpy as np


class MultipleOptimizer(object):
    """
    This class is used to combine multiple optimizers into one.
    This is useful when you have multiple networks that you want to train
    with different learning rates.

    Args:
    op (list): List of Pytorch's optimizers

    Attributes:
        optimizers (list): List of Pytorch's optimizers
    """
    def __init__(self, op):
        self.optimizers = op

    def zero_grad(self, set_to_none=False):
        """
        Sets the gradients to zero for all optimizers
        param set_to_none: If True, sets the gradient to None instead of zero to save some memory
        return: None
        """
        for op in self.optimizers:
            op.zero_grad(set_to_none=set_to_none)

    def step(self, scaler = None):
        """
        Performs an update of the weights for all optimizers
        :param scaler: if given, it uses the scaler to use Pytorch's mixed precision
        :return: None
        """
        for op in self.optimizers:
            if scaler is not None:
                scaler.step(op)
            else:
                op.step()

    def to(self, device):
        """
        Moves the optimizers to the device
        :param device: device to move the optimizers to
        :return: None
        """
        for op in self.optimizers:
            op.to(device)


class MultipleScheduler(object):
    """
    This class is a wrapper for the torch.optim.lr_scheduler.StepLR class.
    It allows for multiple optimizers to be passed in and have their learning
    rates adjusted by the same scheduler.

    Args:
        multioptimizers : MultiOptimizer
            A MultiOptimizer object containing the optimizers to be scheduled.
        step : int
            The number of iterations after which the learning rate is decayed.
        gamma : float
            The factor by which the learning rate is decayed.
    Attributes:
        optimizers: list of optimizers
        schedulers: list of schedulers
    """
    def __init__(self, multioptimizers, step, gamma):
        self.optimizers = multioptimizers.optimizers
        self.schedulers = []

        for op in self.optimizers:
            scheduler = torch.optim.lr_scheduler.StepLR(op, step, gamma=gamma)

            self.schedulers.append(scheduler)

    def step(self):
        """
        Calls the step function on all the schedulers
        :return: None
        """
        for scheduler in self.schedulers:
            scheduler.step()

    def to(self, device):
        """
        Moves the schedulers to the device
        :param device: device to move the schedulers to
        :return: None
        """
        for scheduler in self.schedulers:
            scheduler.to(device)


# Initialization methods
def weight_init(m,actfunc: str = 'sine'):
    """
    Initialize the weights of a layer according to the activation function.

    :param m: torch.nn.Module
        The layer to initialize.
    :param actfunc: str
        The activation function of the layer.
        Options: 'sine', 'relu'.
    :return: None
    """
    if actfunc=='sine':
        with torch.no_grad():
            if hasattr(m, 'weight'):
                num_input = m.weight.size(-1)
                # See supplement Sec. 1.5 for discussion of factor 30
                m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)
                
    elif actfunc=='relu':
            if hasattr(m, 'weight'):
                nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


def first_layer_init(m,actfunc: str ='sine'):
    """
    Initialize the first layer of a network with a uniform/normal distribution.

    :param     m : torch.nn.Module
        The module to be initialized.
    :param actfunc: str
        The activation function of the layer.
        If 'sine', the initialization is done following the uniform distribution.
        If 'relu', the initialization is done with kaiming_normal initialization.
    :return: None
    """
    if actfunc=='sine':
        with torch.no_grad():
            if hasattr(m, 'weight'):
                num_input = m.weight.size(-1)
                m.weight.uniform_(-1 / num_input, 1 / num_input)
    elif actfunc=='relu':
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


class Sine(nn.Module):
    """
    This is a class that implements a sine function.

    It is a subclass of the nn.Module class, which is a class that implements a neural network module.

    It has a single method, forward, which takes an input and returns the sine of 30 times the input.
    """
    def __init(self):
        super().__init__()

    def forward(self, input):
        return torch.sin(30 * input)

                     
class PRTNetwork(nn.Module):
    """
     Class for a PRT (precomputed radiance trasnfer) network.

    Parameters
    ----------
    W : int
        The width of the network.
    D : int
        The depth of the network.
    skips : list
        A list of integers indicating the layers where skip connections are added.
    din : int
        The dimension of the input.
    dout : int
        The dimension of the output.
    activation : str
        The activation function to use.
    """
    def __init__(
            self,           
            W=64,
            D=8,
            skips=[4],
            din=9,
            dout=3,
            activation='relu'
    ):
        super().__init__()
        self.in_features=din
        self.out_features=dout
         
        self.W=W
    
        self.skips=skips
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.in_features,W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.in_features, W) for i in range(D-1)])

        for i, l in enumerate(self.pts_linears):
            if i==0:
                first_layer_init(self.pts_linears[i],activation)
            else:    
                weight_init(self.pts_linears[i],activation)
        
        self.output_linear = nn.Linear(W, self.out_features)
        
        if activation=='sine':
            self.activation= Sine()
     
        elif activation=='relu':
            self.activation=nn.ReLU(inplace=True)
        
    def forward(self,pos,dout,din):
        """
        Returns the model's output, representing the outgoing radiance of points pos
        in directions dout for light coming from directions din.

        The input tensors are assumed to be on the unit sphere.
        :param pos: tensor of encoded 3D positions
        :param dout:  tensor of encoded outgoing directions
        :param din: tensor of encoded ingoing directions
        :return: tensor, output of the neural network
        """
        N=pos.shape[0]
        M=din.shape[0]  
        
        pos=pos.unsqueeze(1).expand([-1,M,-1]) 
        dout=dout.unsqueeze(1).expand([-1,M,-1]) 
        din=din.unsqueeze(0).expand([N,-1,-1])

        all_input=torch.cat([pos, dout,din], dim=-1).view(N*M,-1)
        
        h=all_input
   
        for i, l in enumerate(self.pts_linears):
            
            h = self.pts_linears[i](h)
            h = self.activation(h)
        
            if i in self.skips:
                h = torch.cat([all_input, h], -1)

        output=self.output_linear(h)    

        return output.view(N,M,self.out_features)