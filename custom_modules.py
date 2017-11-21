import math

import torch
import torch.nn as nn
from custom_functions import *


class MIL_max(nn.Module):
    """
        Applies a mil_max transformation to the incoming data: :math:`y = max(x)` at each feature map
    Shape:
        - Input: :math:`(batch_size, channels, in_height, in_width)`
        - Output: :math:`(batch_size, channels, 1,1)`
    Examples::
        >>> input = Variable(torch.rand(100,50,7,7))
        >>> output = mil_max(input)
        >>> output.size()     
    """
    def forward(self, input):
        return mil_max(input)
    def __repr__(self):
        return self.__class__.__name__
		
		
class MIL_or(nn.Module):
    """
        Applies a mil_max transformation to the incoming data: :math:`y = 1-(1-p_{11})...(1-p_{wh})` 
    Shape:
        - Input: :math:`(batch_size, channels, in_height, in_width)`
        - Output: :math:`(batch_size, channels, 1,1)`
    Examples::
        >>> input = Variable(torch.rand(100,50,7,7))
        >>> output = mil_or(input)
        >>> output.size()     
    """
    def forward(self, input):
        return mil_or(input)
    def __repr__(self):
        return self.__class__.__name__ 


class DAG_RNN_se(nn.Module):
    """
        Applies a SouthEast RNN transformation to the incoming data 
    Shape:
        - Input: :math:`(batch_size, channels, in_height, in_width)`
        - Output: :math:`(batch_size, channels,in_height, in_width)`
    Examples::
        >>> input = Variable(torch.rand(100,50,7,7))
        >>> output = dag_rnn_se(input,weight_hh,weight_yh,bias)
        >>> output.size()     
    """
    def __init__(self, input_size, output_size):
        super(DAG_RNN_se, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.weight_hh = nn.Parameter(torch.Tensor(input_size, input_size))
        self.weight_yh = nn.Parameter(torch.Tensor(input_size, output_size))
        self.bias = nn.Parameter(torch.Tensor(output_size))

        # Not a very smart way to initialize weights
        
        #self.weight_hh.data.normal_(0, math.sqrt(2. / n))
        #self.weight_hh.data.normal_(0, math.sqrt(2. / n))
        #self.weight_hh.data.normal_()*1e-3
        self.weight_yh.data.normal_()*1e-3
        self.weight_hh.data = torch.eye(input_size, input_size) 
        self.bias.data.zero_()

    def forward(self, input):
        return dag_rnn_se(input,self.weight_hh,self.weight_yh,self.bias)
    def __repr__(self):
        return self.__class__.__name__ 

class DAG_RNN_sw(nn.Module):
    """
        Applies a SouthWest RNN transformation to the incoming data 
    Shape:
        - Input: :math:`(batch_size, channels, in_height, in_width)`
        - Output: :math:`(batch_size, channels, in_height, in_width)`
    Examples::
        >>> input = Variable(torch.rand(100,50,7,7))
        >>> output = dag_rnn_sw(input,output_lastweight_hh,weight_yh)
        >>> output.size()     
    """
    def __init__(self, input_size, output_size):
        super(DAG_RNN_sw, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weight_hh = nn.Parameter(torch.Tensor(input_size, input_size))
        self.weight_yh = nn.Parameter(torch.Tensor(input_size, output_size))
        # Not a very smart way to initialize weights
        
        #self.weight_hh.data.normal_(0, math.sqrt(2. / n))
        #self.weight_hh.data.normal_(0, math.sqrt(2. / n))
        #self.weight_hh.data.normal_()*1e-3
        self.weight_hh.data = torch.eye(input_size, input_size)
        self.weight_yh.data.normal_()*1e-3

    def forward(self, input, output_last):
        return dag_rnn_sw(input,output_last, self.weight_hh,self.weight_yh)
    def __repr__(self):
        return self.__class__.__name__   
        
class DAG_RNN_nw(nn.Module):
    """
        Applies a NorthWest RNN transformation to the incoming data 
    Shape:
        - Input: :math:`(batch_size, channels, in_height, in_width)`
        - Output: :math:`(batch_size, channels,in_height, in_width)`
    Examples::
        >>> input = Variable(torch.rand(100,50,7,7))
        >>> output = dag_rnn_nw(input,output_last,weight_hh,weight_yh)
        >>> output.size()     
    """
    def __init__(self, input_size, output_size):
        super(DAG_RNN_nw, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weight_hh = nn.Parameter(torch.Tensor(input_size, input_size))
        self.weight_yh = nn.Parameter(torch.Tensor(input_size, output_size))
        # Not a very smart way to initialize weights
        
        #self.weight_hh.data.normal_(0, math.sqrt(2. / n))
        #self.weight_hh.data.normal_(0, math.sqrt(2. / n))
        #self.weight_hh.data.normal_()*1e-3
        self.weight_hh.data = torch.eye(input_size, input_size)
        self.weight_yh.data.normal_()*1e-3

    def forward(self, input, output_last):
        return dag_rnn_nw(input,output_last, self.weight_hh,self.weight_yh)
    def __repr__(self):
        return self.__class__.__name__  
        
class DAG_RNN_ne(nn.Module):
    """
        Applies a NorthEast RNN transformation to the incoming data 
    Shape:
        - Input: :math:`(batch_size, channels, in_height, in_width)`
        - Output: :math:`(batch_size, channels, in_height, in_width)`
    Examples::
        >>> input = Variable(torch.rand(100,50,7,7))
        >>> output = dag_rnn_ne(input,output_last,weight_hh,weight_yh)
        >>> output.size()     
    """
    def __init__(self, input_size, output_size):
        super(DAG_RNN_ne, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weight_hh = nn.Parameter(torch.Tensor(input_size, input_size))
        self.weight_yh = nn.Parameter(torch.Tensor(input_size, output_size))
        # Not a very smart way to initialize weights
        
        #self.weight_hh.data.normal_(0, math.sqrt(2. / n))
        #self.weight_hh.data.normal_(0, math.sqrt(2. / n))
        #self.weight_hh.data.normal_()*1e-3
        self.weight_hh.data = torch.eye(input_size, input_size)
        self.weight_yh.data.normal_()*1e-3

    def forward(self, input, output_last):
        return dag_rnn_ne(input,output_last, self.weight_hh,self.weight_yh)
    def __repr__(self):
        return self.__class__.__name__
