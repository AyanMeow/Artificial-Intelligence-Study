import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from MultiHeadAttention import MultiheadAttention

class TransformerBlock(layers.Layer):
    def __init__(self,hidden_dim,feed_forward_dim,
                 head_num=8,
                 drop_rate=0.1,
                 decoder=False,
                 Kd=None,
                 Vd=None):
        super(TransformerBlock,self).__init__()
        feed_forward_dim=4*hidden_dim
        self.ff_dim=feed_forward_dim
        self.attention=MultiheadAttention(hidden_dim=hidden_dim,
                                          head_num=head_num,
                                          decoder=decoder,
                                          Kd=Kd,Vd=Vd,
                                          )
        self.feed_forward=keras.Sequential(
            [layers.Dense(feed_forward_dim,activation='relu'),
             layers.Dense(hidden_dim,)]
        )
        
        self.normalization1=layers.LayerNormalization(epsilon=1e-6)
        self.normalization2=layers.LayerNormalization(epsilon=1e-6)
        self.dropout1=layers.Dropout(rate=drop_rate)
        self.dropout2=layers.Dropout(rate=drop_rate)
        
    def call(self, inputs,training):
        attention_out=self.attention(inputs)
        #仅在训练时使用dropout
        attention_out=self.dropout1(attention_out,training=training)
        #normalization(x+Attention(x))
        out1=self.normalization1(inputs+attention_out)
        ff_out=self.feed_forward(out1)
        #normalization(x+FeedForward(x))
        ff_out=self.dropout2(ff_out,training=training)
        outputs=self.normalization2(out1+ff_out)
        return outputs
    
#计算PE:直接使用索引    
#maxlen:seq包含的token的最大数量
class PositionAndTokenEmbedding(layers.Layer):
    def __init__(self,hidden_dim,max_len,vocab_size):
        super(PositionAndTokenEmbedding,self).__init__()
        #vocab_size:需要处理的单词序列里不同的单词的总数
        self.token_emb=layers.Embedding(input_dim=vocab_size,
                                        output_dim=hidden_dim,)
        self.pos_emb=layers.Embedding(input_dim=max_len,
                                      output_dim=hidden_dim,)
        self.maxlen=max_len
    
    def call(self,x):
        positions=tf.range(start=0,limit=self.maxlen,delta=1)
        positions=self.pos_emb(positions)
        x_emb=self.token_emb(x)
        return x_emb*tf.math.sqrt(self.hidden_dim)+positions
    
#PE:使用sin cos公式：
class rawPositionAndTokenEmbedding(layers.Layer):
    def __init(self,maxlen,vocab_size,hidden_dim):
        super(rawPositionAndTokenEmbedding,self).__init__()
        self.token_emb=layers.Embedding(input_dim=vocab_size,
                                        output_dim=hidden_dim)
        self.hidden_dim=hidden_dim
        self.maxlen=maxlen
    
    def call(self,inputs):
        x_emb=self.token_emb(inputs)
        pe=tf.zeros(shape=(self.maxlen,self.hidden_dim))
        position=tf.range(start=0,limit=self.maxlen,delta=1)
        #1/10000^(2i/d)=e^(2i*(-(log10000)/d))
        div_term=tf.exp(tf.range(start=0,limit=self.hidden_dim,delta=2) * (-tf.math.log(10000.0)/self.hidden_dim))
        pe[:,0::2]=tf.sin(position*div_term)
        pe[:,1::2]=tf.cos(position*div_term)
        return pe+x_emb*tf.math.sqrt(self.hidden_dim)
            

class TransformModel(object):
    def initModel(self,hidden_dim,ff_dim,maxlen,vocab_size,layer_num=6):
        self.hidden_dim=hidden_dim
        self.ff_dim=ff_dim
        self.maxlen=maxlen
        self.vocab_size=vocab_size
        self.layer_num=layer_num
        self.word_emb=rawPositionAndTokenEmbedding(maxlen,vocab_size,hidden_dim)

    def encoder_stack(self,inputs):
        encoderblock=TransformerBlock(hidden_dim=self.hidden_dim,
                                      head_num=8,
                                      feed_forward_dim=self.ff_dim,
                                      )
        x=inputs
        for _ in range(0,self.layer_num):
            x=encoderblock(x)
        
        return x
    
    def decoder_stack(self,inputs,C):
        #C是encoder输出的编码信息矩阵
        x=inputs
        attention2=TransformerBlock(hidden_dim=self.hidden_dim,
                                    feed_forward_dim=self.ff_dim,
                                    decoder=True,
                                    Kd=C,
                                    Vd=C
                                    )
        for _ in self.layer_num:
            #attention1-masked
            Q=layers.Dense(self.hidden_dim)(x)
            K=layers.Dense(self.hidden_dim)(x)
            V=layers.Dense(self.hidden_dim)(x)
            score=tf.matmul(Q,K,transpose_b=True)
            mask=tf.linalg.band_part(tf.ones_like(score),-1,0)
            score_masked=tf.multiply(score,mask)
            k_dim=tf.cast(tf.shape(K)[-1],tf.float32)
            #根号dk
            scale_score=score_masked/tf.math.sqrt(k_dim)
            #softmax(QK^T/√dk)
            weights=tf.nn.softmax(scale_score)
            #softmax(QK^T/√dk)V
            out1=tf.matmul(weights,V)
            #attention2 使用encoder的输出作为K和V
            out2=attention2(out1)
        
        #Linear
        output_Z=layers.LayerNormalization(epsilon=1e-6)(out2)
        return output_Z
    
    
    def transformer(self,inputs):
        X=self.word_emb(inputs)
        C=self.encoder_stack(X)
        P=self.decoder_stack(X,C)
        O=layers.Dense(self.hidden_dim)(P)
        out=layers.Softmax(axis=-1)(O)
        return out
            