from __future__ import print_function, division
import torch
import torch.nn as nn
import math
from torchvision import models
import torch.nn.functional as F
from custom_functions import *
from custom_modules import *

#__all__ = ['VGG16_FCN','VGG16_MIL', 'VGG16_RNN_concatenate','VGG16_RNN_DAG', 'Resnet50_MIL', 'Resnet50_RNN_concatenate', 'Resnet50_RNN_DAG', 'Resnet101_MIL','Resnet152_MIL','Resnet101_RNN_concatenate', 'Resnet101_RNN_DAG']

__all__ = ['VGG16_FCN','VGG16_MIL', 'Resnet50_MIL', 'Resnet50_RNN_DAG','Resnet101_MIL', 'Resnet152_MIL']
class VGG16_FCN(nn.Module):

    def __init__(self, pretrained, num_classes):
        super(VGG16_FCN, self).__init__()
        tmp_model = models.vgg16(pretrained)
        self.features = tmp_model.features       # copy features
         
        #output = torch.Tensor([crop_height, crop_width]).type(torch.FloatTensor)
        #for _ in xrange(5):
        #    output = torch.ceil(output.div_(2))
        self.fc6 = nn.Conv2d(512,4096,kernel_size=7)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout()
        self.fc7 = nn.Conv2d(4096, 4096,kernel_size=1)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout()
        self.fc8 = nn.Conv2d(4096,num_classes, kernel_size=1)
        
        
        for m in self.fc6, self.fc7:
            self._initialize_weights(m)
        self._initialize_weights_FCN(self.fc8)
        
        # if pretrained:
        #    tmp_parameter = tmp_model.classifier[3].state_dict()
        #    self.fc7.load_state_dict(tmp_parameter)

    def forward(self, x):
        x = self.features(x)
        #x = x.view(x.size(0), -1)
        x = self.fc6(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc7(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc8(x)
        return x
      
    def _initialize_weights(self, m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            n = m.weight.size(1)
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

    def _initialize_weights_FCN(self, m):
        m.weight.data.normal_(0, 0.001)
        if m.bias is not None:
           m.bias.data.fill_(-6.58)

class VGG16_MIL(nn.Module):

    def __init__(self, pretrained, num_classes):
        super(VGG16_MIL, self).__init__()
        tmp_model = models.vgg16(pretrained)
        self.features = tmp_model.features       # copy features
         
        #output = torch.Tensor([crop_height, crop_width]).type(torch.FloatTensor)
        #for _ in xrange(5):
        #    output = torch.ceil(output.div_(2))
        self.fc6 = nn.Conv2d(512,4096,kernel_size=7)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout()
        self.fc7 = nn.Conv2d(4096, 4096,kernel_size=1)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout()
        self.fc8 = nn.Conv2d(4096,num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
        for m in self.fc6, self.fc7:
            self._initialize_weights(m)
        self._initialize_weights_FCN(self.fc8)
        self.mil_or = MIL_or()
        #self.mil_max = MIL_max()
        
        # if pretrained:
        #    tmp_parameter = tmp_model.classifier[3].state_dict()
        #    self.fc7.load_state_dict(tmp_parameter)

    def forward(self, x):
        x = self.features(x)
        #x = x.view(x.size(0), -1)
         
        x = self.fc6(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc7(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc8(x)
        x = self.sigmoid(x)
        x = self.mil_or(x)
        return x
      
    def _initialize_weights(self, m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            n = m.weight.size(1)
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

    def _initialize_weights_FCN(self, m):
        m.weight.data.normal_(0, 0.001)
        if m.bias is not None:
           m.bias.data.fill_(-6.58)

class Resnet50_MIL(nn.Module):

    def __init__(self, pretrained, num_classes):
        super(Resnet50_MIL, self).__init__()
        tmp_model = models.resnet50(pretrained)

        # copy some modules
        self.conv1 = tmp_model.conv1
        self.bn1 = tmp_model.bn1
        self.relu = tmp_model.relu
        self.maxpool = tmp_model.maxpool
        self.layer1 = tmp_model.layer1
        self.layer2 = tmp_model.layer2
        self.layer3 = tmp_model.layer3
        self.layer4 = tmp_model.layer4
        
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Conv2d(tmp_model.fc.in_features, num_classes,kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.mil_or = MIL_or()

        self._initialize_weights_FCN(self.fc)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        #print(x.size())
        x = self.fc(x)
        x = self.sigmoid(x)
        #print(x.size())
        x = self.mil_or(x)
        
        return x

    def _initialize_weights_FCN(self, m):
        m.weight.data.normal_(0, 0.001)
        if m.bias is not None:
           m.bias.data.fill_(-6.58)

class Resnet50_RNN_DAG(nn.Module):

    def __init__(self, pretrained, num_classes):
        super(Resnet50_RNN_DAG, self).__init__()
        tmp_model = models.resnet50(pretrained)

        # copy some modules
        self.conv1 = tmp_model.conv1
        self.bn1 = tmp_model.bn1
        self.relu = tmp_model.relu
        self.maxpool = tmp_model.maxpool
        self.layer1 = tmp_model.layer1
        self.layer2 = tmp_model.layer2
        self.layer3 = tmp_model.layer3
        self.layer4 = tmp_model.layer4
        
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        
        self.transition = nn.Conv2d(tmp_model.fc.in_features, num_classes,kernel_size=1)
        self.dagrnn_se = DAG_RNN_se(num_classes,num_classes) 
        self.dagrnn_sw = DAG_RNN_sw(num_classes,num_classes) 
        self.dagrnn_nw = DAG_RNN_nw(num_classes,num_classes) 
        self.dagrnn_ne = DAG_RNN_ne(num_classes,num_classes)
        #self.bn2 = nn.BatchNorm2d(num_classes)
        self.sigmoid2 = nn.Sigmoid()
        
        #self.fc = nn.Conv2d(tmp_model.fc.in_features, num_classes,kernel_size=1)
        self.fc1 = nn.Conv2d(num_classes, num_classes,kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        '''
        self.transition = nn.Conv2d(num_classes, num_classes,kernel_size=1)
        self.dagrnn_se = DAG_RNN_se(num_classes,num_classes) 
        self.dagrnn_sw = DAG_RNN_sw(num_classes,num_classes) 
        self.dagrnn_nw = DAG_RNN_nw(num_classes,num_classes) 
        self.dagrnn_ne = DAG_RNN_ne(num_classes,num_classes)
        self.softmax = nn.Softmax()
        '''
        self.mil_or = MIL_or()
        self._initialize_weights_Trans(self.transition)
        self._initialize_weights_FCN(self.fc1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        
        x = self.transition(x)
        #print('x')
        #print(x)
        x1 = self.dagrnn_se(x)
        #print('x1')
        #print(x1)
        #x1 = x1.view(x1.size(0), -1)
        #x1 = self.softmax(x1)
        #x1 = x1.view(x.size())
        #print('x1')
        #print(x1)
        x2 = self.dagrnn_sw(x,x1)
        #x2 = x2.view(x2.size(0), -1)
        #x2 = self.softmax(x2)
        #x2 = x2.view(x2.size())
        #print('x2')
        #print(x2)
        x3 = self.dagrnn_nw(x,x2)
        #x3 = x3.view(x3.size(0), -1)
        #x3 = self.softmax(x3)
        #x3 = x3.view(x3.size())
        #print('x3')
        #print(x3)
        x4 = self.dagrnn_ne(x,x3)
        #print('x4')
        #print(x4.size())
        #x4 = self.bn2(x4)
        #print(x4)
        #x4 = x4.view(x4.size(0), -1)
        x5 = self.sigmoid2(x4)
        #x6 = x5.view(x.size())
        #print(x5)
        x = self.fc1(x5)
        x = self.sigmoid(x)
        #print(x)
        x = self.mil_or(x)
        '''
        x = self.fc(x)
        x = self.sigmoid(x)
        x = self.transition(x)
        x1 = self.dagrnn_se(x)
        x2 = self.dagrnn_sw(x,x1)
        x3 = self.dagrnn_nw(x,x2)
        x4 = self.dagrnn_ne(x,x3)
        x4 = x4.view(x4.size(0), -1)
        x5 = self.softmax(x4)
        x6 = x5.view(x.size())
        #x6 = self.sigmoid(x4)
        x = self.mil_or(x6)
        print(x6)
        '''
        return x
    
    def _initialize_weights_Trans(self, m):
        m.weight.data.normal_()*1e-3
        if m.bias is not None:
           m.bias.data.zero_()
    
    def _initialize_weights_FCN(self, m):
        m.weight.data.normal_(0, 0.001)
        if m.bias is not None:
           m.bias.data.fill_(-6.58)

class Resnet101_MIL(nn.Module):

    def __init__(self, pretrained, num_classes):
        super(Resnet101_MIL, self).__init__()
        tmp_model = models.resnet101(pretrained)

        # copy some modules
        self.conv1 = tmp_model.conv1
        self.bn1 = tmp_model.bn1
        self.relu = tmp_model.relu
        self.maxpool = tmp_model.maxpool
        self.layer1 = tmp_model.layer1
        self.layer2 = tmp_model.layer2
        self.layer3 = tmp_model.layer3
        self.layer4 = tmp_model.layer4
        
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        #print(tmp_model.fc.in_features)
        self.fc = nn.Conv2d(tmp_model.fc.in_features, num_classes,kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.mil_or = MIL_or()
          
        self._initialize_weights_FCN(self.fc)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        x = self.mil_or(x)
        
        return x

    def _initialize_weights_FCN(self, m):
        m.weight.data.normal_(0, 0.001)
        if m.bias is not None:
           m.bias.data.fill_(-6.58)


class Resnet152_MIL(nn.Module):

    def __init__(self, pretrained, num_classes):
        super(Resnet152_MIL, self).__init__()
        tmp_model = models.resnet152(pretrained)

        # copy some modules
        self.conv1 = tmp_model.conv1
        self.bn1 = tmp_model.bn1
        self.relu = tmp_model.relu
        self.maxpool = tmp_model.maxpool
        self.layer1 = tmp_model.layer1
        self.layer2 = tmp_model.layer2
        self.layer3 = tmp_model.layer3
        self.layer4 = tmp_model.layer4
        
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        #print(tmp_model.fc.in_features)
        self.fc = nn.Conv2d(tmp_model.fc.in_features, num_classes,kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.mil_or = MIL_or()

        self._initialize_weights_FCN(self.fc)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        x = self.mil_or(x)
        
        return x

    def _initialize_weights_FCN(self, m):
        m.weight.data.normal_(0, 0.001)
        if m.bias is not None:
           m.bias.data.fill_(-6.58)


