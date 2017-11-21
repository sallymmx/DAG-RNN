import math
import numpy as np
import sys
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
from torch.autograd import Variable as Variable
from torch.nn import Module, Parameter

class MIL_MAX(Function):

    def forward(self, input):       
        batch_size, channels, in_height, in_width = input.size()
	prob_min = -sys.float_info.max * torch.ones(batch_size,channels,1,1)
	input_flatten = input.view(batch_size,channels,in_height*in_width)
	input_max_,ind = torch.max(input_flatten,2)
	input_max = input_max_.view(batch_size,channels,1,1)
	output = torch.max(input_max,prob_min)		
	self.save_for_backward(input,output)
        return output
		
    def backward(self, grad_output):
        input, output = self.saved_tensors
	batch_size, channels, in_height, in_width = input.size()
        grad_input = tmp = None
	output_expand = output.expand(batch_size, channels, in_height, in_width)
	grad_input = grad_output.clone()		    	
	grad_input = torch.mul(grad_input, torch.eq(output_expand,input).type(torch.FloatTensor))
	
        return grad_input

		
class MIL_OR(Function):
    def forward(self, input, ):       
        #print(input)
        batch_size, channels, in_height, in_width = input.size()
	prob_min = -sys.float_info.max * torch.ones(batch_size,channels,1,1)
	input_flatten = input.view(batch_size,channels,in_height*in_width)
	input_max_,ind = torch.max(input_flatten,2)
	input_max = input_max_.view(batch_size,channels,1,1)
        nor = 1. - input_flatten
	nor_ = nor.prod(2)
	prod = 1. - nor_.view(batch_size,channels,1,1)
	output = torch.max(prod,input_max)
	ge_zeros = torch.ge(output,0.0)
        if ge_zeros.sum() < batch_size*channels:
            raise ValueError("mil_prob not >= 0")
        le_ones =  torch.le(output,1.0)
	if le_ones.sum() < batch_size*channels:
	    raise ValueError("mil_prob not <= 1")
		
        self.save_for_backward(input,output)
        return output
		
    def backward(self, grad_output):
        input, output = self.saved_tensors
	batch_size, channels, in_height, in_width = input.size()
        grad_input = tmp = None
	output_expand = output.expand(batch_size, channels, in_height, in_width)
	grad_output_expand = grad_output.expand(batch_size, channels, in_height, in_width)
	grad_input = grad_output_expand.clone()
        #print(input.size())
        #print(output.size())
	tmp = torch.div(1 - output_expand,1 - input)
	#if torch.cuda.is_available():
        grad_input = grad_input.mul(torch.clamp(tmp,max=1.))
        #else:
        #   grad_input = grad_input.mul(torch.min(tmp,1.*torch.ones(batch_size, channels, in_height, in_width)))
	        
        return grad_input



class DAG_RNN_SE(Function):
    def forward(self, input, weight_hh, weight_yh, bias_yh):
    # SE plane 
        #print(input)
        #print(weight_hh)
        batch_size, channels, in_height, in_width = input.size()
        output = input-input
        output = output.view(batch_size, channels, in_height * in_width)
        hidden = input
        #print(hidden)
        for b in range(batch_size):
            for h in range(in_height):
                for w in range(in_width):
                    if h > 0:
                        hidden[b,:,h,w] = hidden[b,:,h,w].unsqueeze(1) + torch.mm(weight_hh,hidden[b,:,h-1,w].unsqueeze(1))
                    if w > 0:
                        hidden[b,:,h,w] = hidden[b,:,h,w].unsqueeze(1) + torch.mm(weight_hh,hidden[b,:,h,w-1].unsqueeze(1))   
                    #print(hidden[b,:,h,w])
                    #relu
                    #hidden[b,:,h,w] = torch.clamp(hidden[b,:,h,w],min=0)   
                    hidden[b,:,h,w] = hidden[b,:,h,w].tanh()
                hidden_b = hidden[b,:,:,:]
                hidden_b = hidden_b.view(-1,in_height*in_width) 
                #print(bias_yh.size())         
                output_b =  torch.mm(weight_yh, hidden_b) + bias_yh.unsqueeze(1).expand_as(hidden_b)
                output[b,:,:] = output_b
        #print(output.size())   
        self.save_for_backward(hidden, weight_hh, weight_yh, bias_yh)
        return output  
    
    def backward(self, grad_output):
        
        hidden, weight_hh, weight_yh, bias_yh = self.saved_tensors        
        batch_size, channels, in_height, in_width = hidden.size()
        grad_output_ = grad_output.view(batch_size, -1, in_height * in_width)
        grad_output = grad_output.view(batch_size, -1, in_height, in_width)
        hidden_ = hidden.view(batch_size, channels, in_height * in_width)
        grad_weight_yh = weight_yh - weight_yh
        grad_bias_yh = bias_yh - bias_yh
        grad_weight_hh = weight_hh - weight_hh
        grad_hidden = hidden - hidden
        for b in range(batch_size):
            grad_weight_yh = grad_weight_yh + grad_output_[b,:,:].mm(hidden_[b,:,:].t())
            grad_bias_yh = grad_bias_yh + grad_output_[b,:,:].sum(1)
            for h in range(in_height-1,-1,-1):
                for w in range(in_width-1,-1,-1):
                    grad_hidden[b,:,h,w] = grad_hidden[b,:,h,w].unsqueeze(1) + weight_yh.t().mm(grad_output[b,:,h,w].unsqueeze(1))
                    #grad_hf = grad_hidden[b,:,h,w].mul(grelu(hidden[b,:,h,w]))
                    
                    grad_hf = grad_hidden[b,:,h,w].mul(gtanh(hidden[b,:,h,w]))
                    grad_hh = weight_hh.t().mm(grad_hf.unsqueeze(1))
                    if h > 0:
                        grad_hidden[b,:,h-1,w] = grad_hidden[b,:,h-1,w] + grad_hh
                        grad_weight_hh = grad_weight_hh + grad_hf.unsqueeze(1).mm(grad_hidden[b,:,h-1,w].unsqueeze(1).t())
                    if w > 0:
                        grad_hidden[b,:,h,w-1] = grad_hidden[b,:,h,w-1] + grad_hh
                        grad_weight_hh = grad_weight_hh + grad_hf.unsqueeze(1).mm(grad_hidden[b,:,h,w-1].unsqueeze(1).t())               
        grad_weight_hh_norm = torch.norm(grad_weight_hh.view(-1))
        grad_weight_yh_norm = torch.norm(grad_weight_yh.view(-1))
        grad_bias_yh_norm = torch.norm(grad_bias_yh.view(-1))
        if grad_weight_hh_norm > 2000.0:
            grad_weight_hh = grad_weight_hh / grad_weight_hh_norm * 2000
        if grad_weight_yh_norm > 2000.0:
            grad_weight_yh = grad_weight_yh / grad_weight_yh_norm * 2000
        if grad_bias_yh_norm > 2000.0:
            grad_bias_yh = grad_bias_yh / grad_bias_yh_norm * 2000 
        #print(grad_weight_hh_norm)
        #print(grad_weight_yh_norm)
        #print(grad_bias_yh_norm)
        
        return grad_hidden, grad_weight_hh, grad_weight_yh, grad_bias_yh
    

    
class DAG_RNN_SW(Function):
    def forward(self, input, output_last, weight_hh, weight_yh): 
    # SW Plane
        batch_size, channels, in_height, in_width = input.size()
        output = (input - input).view(batch_size, channels, in_height*in_width)
        hidden = input
        for b in range(batch_size):
                for h in range(in_height-1,-1,-1):
                     for w in range(in_width):
                        if h < in_height-1:
                             hidden[b,:,h,w] = hidden[b,:,h,w].unsqueeze(1) + torch.mm(weight_hh,hidden[b,:,h+1,w].unsqueeze(1))
                        if w > 0:
                             hidden[b,:,h,w] = hidden[b,:,h,w].unsqueeze(1) + torch.mm(weight_hh,hidden[b,:,h,w-1].unsqueeze(1))
                        
                        #hidden[b,:,h,w] = torch.clamp(hidden[b,:,h,w],min=0)
                        hidden[b,:,h,w] = hidden[b,:,h,w].tanh()
                hidden_b = hidden[b,:,:,:]
                hidden_b = hidden_b.view(-1,in_height*in_width)          
                output_b =  torch.mm(weight_yh, hidden_b)
                #print(hidden_b)
                output[b,:,:] = torch.mm(weight_yh, hidden_b)
                output[b,:,:] = output_b
        output = output + output_last
        self.save_for_backward(hidden, weight_hh, weight_yh)        
        return output
    
    def backward(self, grad_output):
        
        hidden, weight_hh, weight_yh = self.saved_tensors        
        batch_size, channels, in_height, in_width = hidden.size()
        grad_output_ = grad_output.view(batch_size, -1, in_height * in_width)
        grad_output = grad_output.view(batch_size, -1, in_height, in_width)
        hidden_ = hidden.view(batch_size, channels, in_height * in_width)
        grad_weight_yh = weight_yh - weight_yh
        grad_weight_hh = weight_hh - weight_hh
        grad_hidden = hidden - hidden
        for b in range(batch_size):
            grad_weight_yh = grad_weight_yh + grad_output_[b,:,:].mm(hidden_[b,:,:].t())
            for h in range(in_height):
                for w in range(in_width-1,-1,-1):
                    grad_hidden[b,:,h,w] = grad_hidden[b,:,h,w].unsqueeze(1) + weight_yh.t().mm(grad_output[b,:,h,w].unsqueeze(1))
                    #grad_hf = grad_hidden[b,:,h,w].mul(grelu(hidden[b,:,h,w]))
                    grad_hf = grad_hidden[b,:,h,w].mul(gtanh(hidden[b,:,h,w]))
                    grad_hh = weight_hh.t().mm(grad_hf.unsqueeze(1))
                    if h < in_height - 1:
                        grad_hidden[b,:,h+1,w] = grad_hidden[b,:,h+1,w] + grad_hh
                        grad_weight_hh = grad_weight_hh + grad_hf.unsqueeze(1).mm(grad_hidden[b,:,h+1,w].unsqueeze(1).t())
                    if w > 0:
                        grad_hidden[b,:,h,w-1] = grad_hidden[b,:,h,w-1] + grad_hh
                        grad_weight_hh = grad_weight_hh + grad_hf.unsqueeze(1).mm(grad_hidden[b,:,h,w-1].unsqueeze(1).t())               
        grad_weight_hh_norm = torch.norm(grad_weight_hh.view(-1))
        grad_weight_yh_norm = torch.norm(grad_weight_yh.view(-1))
        
        if grad_weight_hh_norm > 2000.0:
            grad_weight_hh = grad_weight_hh / grad_weight_hh_norm * 2000
        if grad_weight_yh_norm > 2000.0:
            grad_weight_yh = grad_weight_yh / grad_weight_yh_norm * 2000
        return grad_hidden, None,grad_weight_hh, grad_weight_yh

class DAG_RNN_NW(Function):
    def forward(self, input, output_last, weight_hh, weight_yh,bias_yh=None): 
    # NW Plane
        batch_size, channels, in_height, in_width = input.size()
        output = (input - input).view(batch_size, channels, in_height*in_width) 
        hidden = input
        for b in range(batch_size):
                for h in range(in_height-1,-1,-1):
                     for w in range(in_width-1,-1,-1):
                        if h < in_height-1:
                             hidden[b,:,h,w] = hidden[b,:,h,w].unsqueeze(1) + torch.mm(weight_hh,hidden[b,:,h+1,w].unsqueeze(1))
                        if w < in_width-1:
                             hidden[b,:,h,w] = hidden[b,:,h,w].unsqueeze(1) + torch.mm(weight_hh,hidden[b,:,h,w+1].unsqueeze(1))
                        
                        #hidden[b,:,h,w] = torch.clamp(hidden[b,:,h,w],min=0)
                        hidden[b,:,h,w] = hidden[b,:,h,w].tanh()
                hidden_b = hidden[b,:,:,:]
                hidden_b = hidden_b.view(-1,in_height*in_width)          
                output_b =  torch.mm(weight_yh, hidden_b)
                output[b,:,:] = torch.mm(weight_yh, hidden_b)
                output[b,:,:] = output_b    
        output = output + output_last
        self.save_for_backward(hidden, weight_hh, weight_yh)        
        return output

    def backward(self, grad_output):
        
        hidden, weight_hh, weight_yh = self.saved_tensors        
        batch_size, channels, in_height, in_width = hidden.size()
        grad_output_ = grad_output.view(batch_size, -1, in_height * in_width)
        grad_output = grad_output.view(batch_size, -1, in_height, in_width)
        hidden_ = hidden.view(batch_size, channels, in_height * in_width)
        grad_weight_yh = weight_yh - weight_yh
        grad_weight_hh = weight_hh - weight_hh
        grad_hidden = hidden - hidden
        for b in range(batch_size):
            grad_weight_yh = grad_weight_yh + grad_output_[b,:,:].mm(hidden_[b,:,:].t())
            for h in range(in_height):
                for w in range(in_width):
                    grad_hidden[b,:,h,w] = grad_hidden[b,:,h,w].unsqueeze(1) + weight_yh.t().mm(grad_output[b,:,h,w].unsqueeze(1))
                    #grad_hf = grad_hidden[b,:,h,w].mul(grelu(hidden[b,:,h,w]))
                    grad_hf = grad_hidden[b,:,h,w].mul(gtanh(hidden[b,:,h,w]))
                    grad_hh = weight_hh.t().mm(grad_hf.unsqueeze(1))
                    if h < in_height - 1:
                        grad_hidden[b,:,h+1,w] = grad_hidden[b,:,h+1,w] + grad_hh
                        grad_weight_hh = grad_weight_hh + grad_hf.unsqueeze(1).mm(grad_hidden[b,:,h+1,w].unsqueeze(1).t())
                    if w < in_width - 1:
                        grad_hidden[b,:,h,w+1] = grad_hidden[b,:,h,w+1] + grad_hh
                        grad_weight_hh = grad_weight_hh + grad_hf.unsqueeze(1).mm(grad_hidden[b,:,h,w+1].unsqueeze(1).t())              
        grad_weight_hh_norm = torch.norm(grad_weight_hh.view(-1))
        grad_weight_yh_norm = torch.norm(grad_weight_yh.view(-1))
        
        if grad_weight_hh_norm > 2000.0:
            grad_weight_hh = grad_weight_hh / grad_weight_hh_norm * 2000
        if grad_weight_yh_norm > 2000.0:
            grad_weight_yh = grad_weight_yh / grad_weight_yh_norm * 2000
        return grad_hidden, None, grad_weight_hh, grad_weight_yh    
        
        
        
class DAG_RNN_NE(Function):
    def forward(self, input, output_last, weight_hh, weight_yh): 
    # NE Plane
        batch_size, channels, in_height, in_width = input.size()
        output = (input - input).view(batch_size, channels, in_height*in_width) 
        hidden = input
        for b in range(batch_size):
                for h in range(in_height):
                     for w in range(in_width-1,-1,-1):
                        if h > 0:
                             hidden[b,:,h,w] = hidden[b,:,h,w].unsqueeze(1) + torch.mm(weight_hh,hidden[b,:,h-1,w].unsqueeze(1))
                        if w < in_width-1:
                             hidden[b,:,h,w] = hidden[b,:,h,w].unsqueeze(1) + torch.mm(weight_hh,hidden[b,:,h,w+1].unsqueeze(1))
                        
                        #hidden[b,:,h,w] = torch.clamp(hidden[b,:,h,w],min=0)
                        hidden[b,:,h,w] = hidden[b,:,h,w].tanh()
                hidden_b = hidden[b,:,:,:]
                hidden_b = hidden_b.view(-1,in_height*in_width)          
                output_b =  torch.mm(weight_yh, hidden_b)
                output[b,:,:] = torch.mm(weight_yh, hidden_b)
                output[b,:,:] = output_b    
        output = output + output_last
        #output = F.softmax(output)
        output = output.view(batch_size, channels, in_height, in_width)
        self.save_for_backward(hidden, weight_hh, weight_yh)            
        return output        
    
    def backward(self, grad_output):
        
        hidden, weight_hh, weight_yh = self.saved_tensors        
        batch_size, channels, in_height, in_width = hidden.size()
        grad_output_ = grad_output.view(batch_size, -1, in_height * in_width)
        grad_output = grad_output.view(batch_size, -1, in_height, in_width)
        hidden_ = hidden.view(batch_size, channels, in_height * in_width)
        grad_weight_yh = weight_yh - weight_yh
        grad_weight_hh = weight_hh - weight_hh
        grad_hidden = hidden - hidden
        for b in range(batch_size):
            grad_weight_yh = grad_weight_yh + grad_output_[b,:,:].mm(hidden_[b,:,:].t())
            for h in range(in_height-1,-1,-1):
                for w in range(in_width):
                    grad_hidden[b,:,h,w] = grad_hidden[b,:,h,w].unsqueeze(1) + weight_yh.t().mm(grad_output[b,:,h,w].unsqueeze(1))
                    #grad_hf = grad_hidden[b,:,h,w].mul(grelu(hidden[b,:,h,w]))
                    grad_hf = grad_hidden[b,:,h,w].mul(gtanh(hidden[b,:,h,w]))
                    grad_hh = weight_hh.t().mm(grad_hf.unsqueeze(1))
                    if h > 0:
                        grad_hidden[b,:,h-1,w] = grad_hidden[b,:,h-1,w] + grad_hh
                        grad_weight_hh = grad_weight_hh + grad_hf.unsqueeze(1).mm(grad_hidden[b,:,h-1,w].unsqueeze(1).t())
                    if w < in_width - 1:
                        grad_hidden[b,:,h,w+1] = grad_hidden[b,:,h,w+1] + grad_hh
                        grad_weight_hh = grad_weight_hh + grad_hf.unsqueeze(1).mm(grad_hidden[b,:,h,w+1].unsqueeze(1).t())               
        grad_weight_hh_norm = torch.norm(grad_weight_hh.view(-1))
        grad_weight_yh_norm = torch.norm(grad_weight_yh.view(-1))
        
        if grad_weight_hh_norm > 2000.0:
            grad_weight_hh = grad_weight_hh / grad_weight_hh_norm * 2000
        if grad_weight_yh_norm > 2000.0:
            grad_weight_yh = grad_weight_yh / grad_weight_yh_norm * 2000
        return grad_hidden, None, grad_weight_hh, grad_weight_yh  


def mil_max(input):

    return MIL_MAX()(input)

def mil_or(input):

    return MIL_OR()(input)

def grelu(input):
    ginput = input.clamp(min=0.0)
    ginput = ginput.gt(0).type_as(input)
    return ginput
def gtanh(input):
    ginput = 1 - input.mul(input)
    return ginput

def dag_rnn_se(input,weight_hh, weight_yh, bias_yh):
    return DAG_RNN_SE()(input,weight_hh, weight_yh, bias_yh)

def dag_rnn_sw(input,output_last, weight_hh, weight_yh):
    return DAG_RNN_SW()(input,output_last, weight_hh, weight_yh)
    
def dag_rnn_nw(input,output_last, weight_hh, weight_yh):
    return DAG_RNN_NW()(input,output_last, weight_hh, weight_yh)

def dag_rnn_ne(input,output_last, weight_hh, weight_yh):
    return DAG_RNN_NE()(input, output_last, weight_hh, weight_yh)

# simple test
if __name__ == "__main__":
    from torch.autograd import Variable

    torch.manual_seed(1111)
    a = torch.randn(4,3,2, 3)
    a = torch.min(a,1*torch.ones(4,3,2,3))
    a = torch.max(a,0.001+torch.zeros(4,3,2,3))
    a = torch.sigmoid(a)
    print(a)    
    va = Variable(a, requires_grad=True)
    weight_hh = Parameter(torch.rand(3, 3))
    weight_yh = Parameter(torch.rand(3, 3))
    bias_yh = Parameter(torch.rand(3))
    #print(va)
    
    vb = dag_rnn_se(va, weight_hh, weight_yh, bias_yh)
    vc = dag_rnn_sw(va,vb, weight_hh, weight_yh) 
    vd = dag_rnn_nw(va,vc, weight_hh, weight_yh)
    ve = dag_rnn_ne(va,vd, weight_hh, weight_yh) 
    print vb.data,vc.data,vd.data,ve.data

    ve.backward(torch.ones(va.size()))
    print ve.grad.data
