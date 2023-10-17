import torch.nn as nn
import torch 
import numpy as np


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
     
     def forward(self,inputs):
        '''slidewindow from right
        input:(N,128)
        '''
        inputs=inputs.to(self.device)
        seq_len=inputs.size(0)
        while seq_len>1:
            father_temp=[]
            recon_loss_temp=[]
            for i in range(1,seq_len,1):
                inputs
            
            

    