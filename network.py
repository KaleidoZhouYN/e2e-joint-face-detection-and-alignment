#coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
from collections import OrderedDict
import time
import numpy as np

class Conv2d(nn.Module):
    """self define convolution layer
       Basic structure:
       Convolution + BatchNorm2d + ReLU
       if ReLU == None it becomes a Linear BottleNeck
    """
    def __init__(self,in_channels, out_channels,kernel_size,stride=1,padding=0,dilation=1,group=1,relu=None):
        super(Conv2d, self).__init__()
        if (relu != None):
            if relu == 'relu':
                relu_layer = nn.ReLU(inplace=True)
            if relu == 'prelu':
                relu_layer = nn.PReLU(num_parameters=out_channels)
            self.main = nn.Sequential(OrderedDict([
                ('Conv2d',nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,dilation,group,bias=False)),
                ('BatchNorm',nn.BatchNorm2d(out_channels,affine=True)),
                ('PReLU',relu_layer)
            ]))
        else:
            self.main = nn.Sequential(OrderedDict([
                ('Conv2d',nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,dilation,group,bias=False)),
                ('BatchNorm',nn.BatchNorm2d(out_channels,affine=True))
            ]))
    
    def forward(self,x):
        return self.main(x)
        
        
class ResidualBlock(nn.Module):
    """Residual Block
       t:expand factor in Residual      
    """
    def __init__(self, dim_in,dim_out,stride=1,kernel_size=3,t=1,padding=1):
        super(ResidualBlock, self).__init__()
        self.stride = stride
        if stride == 1:
            od = OrderedDict([
                ('Conv2d_1',Conv2d(in_channels=dim_in,out_channels=dim_in*t,kernel_size=kernel_size,stride=1,padding=padding,relu='prelu')),
                ('Conv2d_2',Conv2d(in_channels=dim_in*t,out_channels=dim_out,kernel_size=kernel_size,stride=1,padding=padding))
                ])
            self.main = nn.Sequential(od)
        else:
            od = OrderedDict([
                ('Conv2d_1',Conv2d(in_channels=dim_in,out_channels=dim_in*t,kernel_size=kernel_size,stride=1,padding=padding,relu='prelu')),
                ('Conv2d_2',Conv2d(in_channels=dim_in*t,out_channels=dim_out,kernel_size=kernel_size,stride=2,padding=padding))
                ])
            self.main = nn.Sequential(od)
            self.short = Conv2d(in_channels=dim_in,out_channels=dim_out,kernel_size=1,stride=2,padding=0,relu='prelu')               
                
    def forward(self,x):
        if self.stride == 1:
            return x + self.main(x)
        else:
            return self.short(x) + self.main(x)
        

class BoxLayer(nn.Module):
    num_classes = 2
    num_anchors = [1]    
    def __init__(self,in_planes=[128]):
        super(BoxLayer,self).__init__()
        self.in_planes = in_planes
        self.loc_layers = nn.ModuleList()
        self.conf_layers = nn.ModuleList()
        self.ldmk_layers = nn.ModuleList()
        for i in range(len(self.in_planes)):
            self.loc_layers.append(Conv2d(self.in_planes[i],self.num_anchors[i]*4,3,1,1))
            self.conf_layers.append(Conv2d(self.in_planes[i],self.num_anchors[i]*2,3,1,1))

            
    def forward(self,xs):
        '''
        xs:list of feature map
        return: loc_preds: [N,-,4]
                conf_preds:[N,-,2]
        '''
        y_locs = []
        y_confs = []
        for i,x in enumerate(xs):
            y_loc = self.loc_layers[i](x)
            N = y_loc.size(0)
            y_loc = y_loc.permute(0,2,3,1).contiguous()
            y_loc = y_loc.view(N,-1,4)
            y_locs.append(y_loc)
            
            y_conf = self.conf_layers[i](x)
            y_conf = y_conf.permute(0,2,3,1).contiguous()
            y_conf = y_conf.view(N,-1,2)
            y_confs.append(y_conf)
            
        loc_preds = torch.cat(y_locs,1)
        conf_preds = torch.cat(y_confs,1)
                
        return loc_preds, conf_preds

class LdmkLayer(nn.Module):
    num_anchors = [1]
    def __init__(self,in_planes=[128]):
        super(LdmkLayer,self).__init__()
        self.in_planes = in_planes
        self.ldmk_layers = nn.ModuleList()
        for i in range(len(self.in_planes)):
            self.ldmk_layers.append(Conv2d(self.in_planes[i],self.num_anchors[i]*10,3,1,1))
            
    def forward(self,xs):
        '''
        xs:list of feature map
        return: ldmk_preds:[N,-,10]
        '''
        y_ldmks = []
        for i,x in enumerate(xs):
            y_ldmk = self.ldmk_layers[i](x) # N,anchors*10,H,W
            N = y_ldmk.size(0)
            y_ldmk = y_ldmk.permute(0,2,3,1).contiguous()
            y_ldmk = y_ldmk.view(N,-1,10) # N,anchors[i]*H*W,10
            y_ldmks.append(y_ldmk)
        
        ldmk_preds = torch.cat(y_ldmks,1)   

        return ldmk_preds
    
class PNet(nn.Module):
    """
    """
    def __init__(self,c=[16,24,32,48],resblock=[0,1,3,2],t=2):
        super(PNet,self).__init__()
        self.down1 = Conv2d(3,c[0],3,2,1,relu='prelu') # 32x32 RF = 3x3 , s = 2
        
        res2 = collections.OrderedDict()
        for i in xrange(resblock[0]):
            res2[str(i)] = ResidualBlock(dim_in=c[0],dim_out=c[0],stride=1,t=t)
        self.res2 = nn.Sequential(res2)        
        self.down2 = Conv2d(c[0],c[1],3,2,1,relu='prelu') # 16x16 RF = 7 , s = 4
        
        res3 = collections.OrderedDict()
        for i in xrange(resblock[1]):
            res3[str(i)] = ResidualBlock(dim_in=c[1],dim_out=c[1],stride=1,t=t)
        self.res3 = nn.Sequential(res3)        
        self.down3 = Conv2d(c[1],c[2],3,2,1,relu='prelu') # 8x8 , RF = 15 , s = 8
        
        res4 = collections.OrderedDict()
        for i in xrange(resblock[2]): 
            res4[str(i)] = ResidualBlock(dim_in=c[2],dim_out=c[2],stride=1,t=t)
        self.res4 = nn.Sequential(res4)        
        self.down4 = Conv2d(c[2],c[3],3,2,1,relu='prelu') # 4x4 , RF = 31, s=16

        res5 = collections.OrderedDict()
        for i in xrange(resblock[3]):
            res5[str(i)] = ResidualBlock(dim_in=c[3],dim_out=c[3],stride=1,t=t)
        self.res5 = nn.Sequential(res5)
        
        fm = [c[3]]
        self.box = BoxLayer(in_planes=fm)
        
    def forward(self,x,mode=0):
        down1 = self.down1(x)
        down2 = self.down2(self.res2(down1))
        down3 = self.down3(self.res3(down2))
        down4 = self.down4(self.res4(down3))
        res5 = self.res5(down4)
        hs = [res5]
        
        loc_preds, conf_preds = self.box(hs)
        return loc_preds, conf_preds     
        
        
class ONet(nn.Module):
    """
    """
    def __init__(self,c=[16,32,32,16],resblock=[2,3,1,0],t=2):
        super(ONet,self).__init__()
        self.c = c
        self.resblock = resblock
        self.conv1 = Conv2d(3,c[0]*t,3,1,1,relu='prelu')
        self.down1 = Conv2d(c[0]*t,c[0],3,2,1,relu='prelu') # s= 2
        
        res2 = collections.OrderedDict()
        for i in xrange(resblock[0]):
            res2[str(i)] = ResidualBlock(dim_in=c[0],dim_out=c[0],stride=1,t=t)
        self.res2 = nn.Sequential(res2)
        self.down2 = Conv2d(c[0],c[1],3,2,1,relu='prelu') # s = 4
        
        res3 = collections.OrderedDict()
        for i in xrange(resblock[1]):
            res3[str(i)] = ResidualBlock(dim_in=c[1],dim_out=c[1],stride=1,t=t)
        self.res3 = nn.Sequential(res3)
        self.down3 = Conv2d(c[1],c[2],3,2,1,relu='prelu') # s = 8
        
        res4 = collections.OrderedDict()
        for i in xrange(resblock[2]):
            res4[str(i)] = ResidualBlock(dim_in=c[2],dim_out=c[2],stride=1,t=t)
        self.res4 = nn.Sequential(res4)    
        self.down4 = Conv2d(c[2],c[3],3,2,1,relu='prelu') # s = 16
        
        res5 = collections.OrderedDict()
        for i in xrange(resblock[3]):
            res5[str(i)] = ResidualBlock(dim_in=c[3],dim_out=c[4],stride=1,t=t)
        self.res5 = nn.Sequential(res5)
        
        fm=[c[3]]
        self.ldmk = LdmkLayer(in_planes=fm)
        
    def forward(self,x):
        down1 = self.down1(self.conv1(x))
        down2 = self.down2(self.res2(down1))
        down3 = self.down3(self.res3(down2))
        down4 = self.down4(self.res4(down3))
        res5 = self.res5(down4)
        hs = [res5]
        
        ldmk_preds = self.ldmk(hs)
        return ldmk_preds
        
if __name__ == '__main__':
    onet = ONet()    
    pnet = PNet(c=[16,24,32,48],resblock=[0,1,3,2])
    data = torch.randn(1,3,1024,1024)
    torch.set_num_threads(20)
    t_t = 0
    for i in xrange(20):
        s_t = time.time()
        pnet(data)
        e_t = time.time()
        t_t += e_t - s_t
        print(e_t-s_t)
    print('avg',t_t/20)