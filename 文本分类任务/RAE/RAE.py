import torch.nn as nn
import torch.nn.functional as F
import torch 
import numpy as np
from tree import RTree

class RAE(nn.Module):
    """docstring for RAE.
    """
    def __init__(self,
                 device=torch.device('cpu'),
                 emb_dim=128,
                 vocab_size=3000,
                 K=5,
                 ):
        super(RAE, self).__init__()
        self.device=device
        in_size=emb_dim*2
        mid_size=int(np.ceil(emb_dim*1.5))
        self.emb=nn.Embedding(num_embeddings=vocab_size,embedding_dim=emb_dim).to(device)
        self.encoder=nn.Sequential(
            nn.Linear(in_size,mid_size).to(device),
            nn.Linear(mid_size,emb_dim).to(device)
        )
        self.decoder=nn.Sequential(
            nn.Linear(emb_dim,mid_size).to(device),
            nn.Linear(mid_size,in_size).to(device)
        )
        self.soft=nn.Sequential(
            nn.Linear(emb_dim,K).to(device),
            nn.Softmax(dim=-1)
        )
        self.bitree=None
     
    def forward(self,inputs):
        '''slidewindow from right
        input:(b,512)
        '''
        inputs=inputs.to(self.device)
        X=self.emb(inputs)
        #X(b,512,128)
        Bsize,seq_len,m=X.size()
        pred=[]
        Loss=0
        for b in range(0,Bsize):
            nodelist=[]
            for i in range(0,seq_len,1):
                n=RTree(x=X[b][i])
                nodelist.append(n)
            while len(nodelist)>1:
                father_temp=[]
                recon_loss_temp=[]
                recon_loss=[]
                for i in range(len(nodelist)-1,0,-1):
                    #计算父节点和重建损失
                    temp=torch.concat([nodelist[i].x,nodelist[i-1].x])
                    yi=self.encoder(temp)
                    xx=self.decoder(yi)
                    loss1=F.cross_entropy(xx[0:m],nodelist[i].x)
                    loss2=F.cross_entropy(xx[m:2*m],nodelist[i-1].x)
                    #计算加权loss和
                    nodelist[i].cal_leaves()
                    nodelist[i-1].cal_leaves()
                    n1=nodelist[i].leave_count
                    n2=nodelist[i-1].leave_count
                    loss=(n1+1e-5)/(n1+n2+1e-5)*loss1+(n2+1e-5)/(n1+n2+1e-5)*loss2
                    father_temp.insert(0,yi)
                    recon_loss_temp.insert(0,loss.cpu().detach().numpy())
                    recon_loss.insert(0,loss)
                #贪心算法，挑选最小重建损失
                idx=np.argmin(recon_loss_temp)
                Loss+=recon_loss[idx]
                father=RTree(x=father_temp[idx],
                            K=self.soft(father_temp[idx]),
                            loss=recon_loss_temp[idx],
                            lc=nodelist[idx-1],
                            rc=nodelist[idx])
                nodelist.insert(idx,father)
                #替换原本子节点
                nodelist.remove(father.lc)
                nodelist.remove(father.rc)   
            self.bitree=nodelist[0]
            pred.append(torch.argmax(self.bitree.K))
        pred=torch.from_numpy(np.array(pred))
        return pred,Loss/Bsize
            

    