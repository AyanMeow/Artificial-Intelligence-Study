import torch.nn as nn
import torch.nn.functional as F
import torch 
import numpy as np
from Tree import RTree

class RAE(nn.Module):
    """docstring for RAE.
    """
    def __init__(self,
                 device,
                 input_dim,
                 K,
                 ):
        super(RAE, self).__init__()
        self.device=device
        self.encoder=nn.Sequential(
            nn.Linear(256,200).to(device),
            nn.Linear(200,128).to(device)
        )
        self.decoder=nn.Sequential(
            nn.Linear(128,200).to(device),
            nn.Linear(200,256).to(device)
        )
        self.bitree=None
     
    def forward(self,inputs):
        '''slidewindow from right
        input:(N,128)
        '''
        inputs=inputs.to(self.device)
        seq_len,m=inputs.size()
        nodelist=[]
        for i in range(0,seq_len,1):
            n=RTree(x=inputs[i])
            nodelist.append(n)
        while len(nodelist)>1:
            father_temp=[]
            recon_loss_temp=[]
            for i in range(len(nodelist)-1,0,-1):
                #计算父节点和重建损失
                temp=torch.concat([nodelist[i].x,nodelist[i-1].x])
                yi=self.encoder(temp)
                xx=self.decoder(yi)
                loss1=F.cross_entropy(nodelist[i].x,xx[0:m])
                loss2=F.cross_entropy(nodelist[i-1].x,xx[m:-1])
                #计算加权loss和
                nodelist[i].cal_leaves()
                nodelist[i-1].cal_leaves()
                n1=nodelist[i].leave_count
                n2=nodelist[i-1].leave_count
                loss=n1/(n1+n2)*loss1+n2/(n1+n2)*loss2
                
            

    