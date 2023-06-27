import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class MultiheadAttention(layers.Layer):
    def __init(self,hidden_dim,head_num=8,decoder=False,Kd=None,Vd=None):
        super(MultiheadAttention,self).__init__()
        self.embed_num=hidden_dim
        self.head_num=head_num
        if hidden_dim % head_num != 0 :
            raise ValueError(
                f"embed_dim is not divisiable."
            )
        #？
        self.head_size=hidden_dim/head_num
        #QKV的维度
        self.Q_dens=layers.Dense(hidden_dim)
        self.K_dens=layers.Dense(hidden_dim)
        self.V_dens=layers.Dense(hidden_dim)
        #concat之后的linear层
        self.combine_heads=layers.Dense(hidden_dim)
        self.decoder=decoder
        self.Kd=Kd
        self.Vd=Vd
        
        
    def self_attention(self,q,k,v):
        #QK^T
        score=tf.matmul(q,k,transpose_b=True)
        #取K最后一维，即向量列数维度,dk
        k_dim=tf.cast(tf.shape(k)[-1],tf.float32)
        #根号dk
        scale_score=score/tf.math.sqrt(k_dim)
        #softmax(QK^T/√dk)
        weights=tf.nn.softmax(scale_score)
        #softmax(QK^T/√dk)V
        outputs=tf.matmul(weights,v)
        return outputs,weights
    
    #用矩阵乘法代替循环，这样点积可以看作大小为(head_num,seq_len,head_size)和(head_num,head_size,seq_len)的两个张量相乘，
    #得到一个(head_num,seq_len,seq_len)的矩阵，其实就相当于(seq_len,head_size)和(head_size,seq_len)的两个矩阵相乘，做了head_num次
    def separate_heads(self,x,batch_size):
        x=tf.reshape(x,(batch_size,-1,self.head_num,self.head_size))
        return tf.transpose(x,perm=[0,2,1,3])
    
    def call(self,inputs):
        #inputs=(b,sep_len,embedding_dim) (b,n,d)
        batch_size=tf.shape(inputs)[0]
        if self.decoder:
            Q=self.Q_dens(inputs)
            K=self.Kd
            V=self.Vd
        else:
            Q=self.Q_dens(inputs)
            K=self.K_dens(inputs)
            V=self.V_dens(inputs)
        Q=self.separate_heads(x=Q,batch_size=batch_size)
        #Q,K,V:(batch_size,head_num,seq_len,head_size)
        K=self.separate_heads(x=K,batch_size=batch_size)
        V=self.separate_heads(x=V,batch_size=batch_size)
        attention,weights=self.self_attention(Q,K,V)
        attention=tf.transpose(attention,perm=[0,2,1,3])
        concat_attention=tf.reshape(attention,(batch_size,-1,self.embed_num))
        output=self.combine_heads(concat_attention)
        return output