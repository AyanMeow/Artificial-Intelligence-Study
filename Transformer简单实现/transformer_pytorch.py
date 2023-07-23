import torch
import math
import torch.nn as nn
import numpy as np


#src_vocab_size: 源词表大小
src_vocab_size=3000
#trg_vocab_size: 目标词表大小
trg_vocab_szie=3000
#d_model: 隐藏维度
d_model=512
#d_ff: 前馈网络隐藏层大小
d_ff=d_model*4
#n_layers: 层数
n_layers=6
#maxlen: 支持最大序列长度
maxlen=5000
#d_k,d_v: 注意力单头隐藏层大小 =d_model/n_heads
d_k,d_v=64,64


class Transformer(nn.Module):
    """transformer"""
    def __init__(self, ):
        super(Transformer, self).__init__()
        self.encoder=Encoder().cuda()
        self.decoder=Decoer().cuda()
        self.projection=nn.Linear(d_model,trg_vocab_szie,bias=False).cuda()
        
    def forward(self,enc_inputs,dec_inputs):
        enc_outputs,enc_attn=self.encoder(enc_inputs)
        dec_outputs,dec_self_attn,dec_enc_attn=self.decoder(dec_inputs,enc_inputs,enc_outputs)
        dec_logits=self.projection(dec_outputs)
        return dec_logits.view(-1,dec_logits.size(-1)),enc_attn,dec_self_attn,dec_enc_attn

    

    

class Encoder(nn.Module):
    def __init__(self,) -> None:
        super(self,Encoder).__init__()
        self.embedding=nn.Embedding(src_vocab_size,d_model)
        self.positionemb=PositionEmb(d_model=d_model)
        self.layers=nn.ModuleList(
            [EncoderLayer() for _ in n_layers]
        )
    
    def forward(self,enc_inputs):
        enc_outputs=self.embedding(enc_inputs)
        enc_outputs=self.positionemb(enc_outputs.transpose(0,1)).transpose(0,1)
        enc_self_attn_mask=get_attn_pad_mask(enc_inputs,enc_inputs)
        enc_self_attn=[]
        for layer in self.layers:
            enc_outputs,self_attn=layer(enc_outputs,enc_self_attn_mask)
            enc_self_attn.append(self_attn)
        return enc_outputs,enc_self_attn
        
class EncoderLayer(nn.Module):
    """docstring for EncoderLayer."""
    def __init__(self, arg):
        super(EncoderLayer, self).__init__()
        self.enc_selfattn=MutiHeadAttention()
        self.feedforward=FeedForwardNet()
        
    def forward(self,enc_inputs,enc_attn_mask):
        enc_outputs,attn=self.enc_selfattn(enc_inputs,enc_inputs,enc_inputs,enc_attn_mask) 
        enc_outputs=self.feedforward(enc_outputs)
        return enc_outputs,attn       
        
class Decoer(nn.Module):
    def __init__(self, ):
        super(Decoer, self).__init__()
        self.trg_emb=nn.Embedding(trg_vocab_szie,d_model)
        self.positionemb=PositionEmb()
        self.layers=nn.ModuleList(
            [DecoderLayer() for _ in n_layers]
        )
    
    def forward(self,dec_inputs,enc_inputs,enc_outputs):
        dec_outputs=self.trg_emb(dec_inputs)
        dec_outputs=self.positionemb(dec_outputs.transpose(0,1)).transpose(0,1).cuda()
        self_attn_mask=get_attn_pad_mask(dec_inputs,dec_inputs).cuda()
        sub_attn_mask=get_sub_attn_pad_mask(dec_inputs).cuda()
        dec_self_mask=torch.gt((self_attn_mask+sub_attn_mask),0).cuda
        dec_enc_mask=get_attn_pad_mask(dec_inputs,enc_inputs)
        
        dec_self_attn,dec_enc_attn=[],[]
        for layer in self.layers:
            dec_outputs,self_attn,de_attn=layer(dec_outputs,enc_outputs,dec_self_mask,dec_enc_mask)
            dec_self_attn.append(self_attn)
            dec_enc_attn.append(de_attn)
        return dec_outputs,dec_self_attn,dec_enc_attn
    

class DecoderLayer(nn.Module):
    """docstring for DecoderLayer."""
    def __init__(self, ):
        super(DecoderLayer, self).__init__()
        self.dec_enc_attn=MutiHeadAttention()
        self.dec_self_attn=MutiHeadAttention()
        self.fdw=FeedForwardNet()
        
    def forward(self,dec_inputs,enc_outputs,dec_self_attn_mask,dec_enc_attn_mask):
        dec_outputs,dec_self_attn=self.dec_self_attn(dec_inputs,dec_inputs,dec_inputs,dec_self_attn_mask)
        dec_outputs,dec_enc_attn=self.dec_enc_attn(dec_outputs,enc_outputs,enc_outputs,dec_enc_attn_mask)
        dec_outputs=self.fdw(dec_outputs)
        return dec_outputs,dec_self_attn,dec_enc_attn

def get_attn_pad_mask(seq_q,seq_k):
    b_size,len_q=seq_q.size()
    b_size,len_k=seq_k.size()

    pad_mask=seq_k.data.eq(0).unsqueeze(1)
    return pad_mask.expand(b_size,len_q,len_k)


def get_sub_attn_pad_mask(seq):
    attn_shape=[seq.size(0),seq.size(1),seq.size(1),]
    subsequence_mask=np.triu(np.ones(attn_shape),k=1)
    subsequence_mask=torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask
        
class PositionEmb(nn.Module):
    """docstring for PositionEmb."""
    def __init__(self, 
                 maxlen=5000,
                 dropout_rate=0.1,
                 ):
        super(PositionEmb, self).__init__()
        self.dropout=nn.Dropout(dropout_rate)
        self.pe=torch.zeros(maxlen,d_model)
        position=torch.arange(0,maxlen,1,dtype=torch.float).unsqueeze(1)
        div_iterm=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        self.pe[:,0::2]=torch.sin(position*div_iterm)
        self.pe[:,1::2]=torch.cos(position*div_iterm)
        self.pe.unsqueeze_(0).transpose_(0,1)
    
    def forward(self,x):
        x=x+self.pe[:x.size(0),:]
        return self.dropout(x)




class MutiHeadAttention(nn.Module):
    """docstring for MutiHeadAttention."""
    def __init__(self,
                 n_heads=8
                 ):
        super(MutiHeadAttention, self).__init__()
        self.n_heads=n_heads
        self.Q_w=nn.Linear(d_model,d_k*n_heads,bias=False)
        self.K_w=nn.Linear(d_model,d_k*n_heads,bias=False)
        self.V_w=nn.Linear(d_model,d_v*n_heads,bias=False)
        self.fc=nn.Linear(d_v*n_heads,d_model,bias=False)
        
    def forward(self,q,k,v,
                attn_mask,
                ):
        #[b,h,n,d_k]
        residual,b_size=q,q.size(0)
        Q=self.Q_w(q).view(b_size,-1,self.n_heads,d_k).transpose(1,2)
        K=self.K_w(k).view(b_size,-1,self.n_heads,d_k).transpose(1,2)
        V=self.V_w(v).view(b_size,-1,self.n_heads,d_v).transpose(1,2)
        
        attn_mask.unsqueeze_(1).repeat(1,self.n_heads,1,1)

        context,attn=ScaleDotProductAttention()(Q,K,V,attn_mask)
        context.transpose_(1,2).reshape_(b_size,-1,d_k*self.n_heads)
        output=self.fc(context)
        
        return nn.LayerNorm(d_model).cuda()(output+residual),attn


class ScaleDotProductAttention(nn.Module):      
    """docstring for ScaleDotProductAttention."""
    def __init__(self,):
        super(ScaleDotProductAttention, self).__init__()

    def forward(self,Q,K,V,att_mask):
        score=torch.matmul(Q,K.transpose(-1,-2))/np.sqrt(d_k)
        score.masked_fill_(att_mask,-1e9)
        
        attn=nn.Softmax(dim=-1)(score)
        context=torch.matmul(attn,V)
        return context,attn        

    

class FeedForwardNet(nn.Module)     :
    """docstring for FeedForwardNet."""
    def __init__(self,):
        super(FeedForwardNet, self).__init__()
        self.fc=nn.Sequential(
            nn.Linear(d_model,d_ff,bias=False),
            nn.ReLU(),
            nn.Linear(d_ff,d_model,bias=False)
        )
        self.layerNorm=nn.LayerNorm(d_model).cuda()
        
    def forward(self,inputs):
        residual=inputs
        outputs=self.fc(inputs)
        return self.layerNorm(outputs+residual)

    
