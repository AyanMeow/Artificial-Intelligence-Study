# Transformer  
整体结构：  
![transformer](https://pic4.zhimg.com/80/v2-4544255f3f24b7af1e520684ae38403f_720w.webp)  
## 1. 工作流程  
* 第一步：获取输入句子的每一个单词的表示向量 X，X由单词的 Embedding（Embedding就是从原始数据提取出来的Feature） 和单词位置的 Embedding 相加得到。  
* 第二步：将得到的单词表示向量矩阵 (如上图所示，每一行是一个单词的表示 x) 传入 Encoder 中，经过 6 个 Encoder block 后可以得到句子所有单词的编码信息矩阵 C，如下图。单词向量矩阵用 $X_{n×d}$ 表示， n 是句子中单词个数，d 是表示向量的维度 (论文中 d=512)。每一个 Encoder block 输出的矩阵维度与输入完全一致。  
* 第三步：将 Encoder 输出的编码信息矩阵 C传递到 Decoder 中，Decoder 依次会根据当前翻译过的单词 1~ i 翻译下一个单词 i+1，如下图所示。在使用的过程中，翻译到单词 i+1 的时候需要通过 Mask (掩盖) 操作遮盖住 i+1 之后的单词。  

## 2. 关于单词embedding 
Transformer 中除了单词的 Embedding，还需要使用位置 Embedding 表示单词出现在句子中的位置。**因为 Transformer 不采用 RNN 的结构，而是使用全局信息，不能利用单词的顺序信息，而这部分信息对于 NLP 来说非常重要。** 所以 Transformer 中使用位置 Embedding 保存单词在序列中的相对或绝对位置。  

位置 Embedding 用 PE表示，PE 的维度与单词 Embedding 是一样的。PE 可以通过训练得到，也可以使用某种公式计算得到。在 Transformer 中采用了后者，计算公式如下：  
$$PE_{(pos,2i)=sin(\frac{pos}{10000^{ 2 i/d}})}$$  
$$PE_{(pos,2i+1)=cos(\frac{pos}{10000^{ 2 i/d}})}$$  
其中，pos 表示单词在句子中的位置，d 表示 PE的维度 (与词 Embedding 一样)，2i 表示偶数的维度，2i+1 表示奇数维度 (即 2i≤d, 2i+1≤d)。使用这种公式计算 PE 有以下的好处：  

* 使 PE 能够适应比训练集里面所有句子更长的句子，假设训练集里面最长的句子是有 20 个单词，突然来了一个长度为 21 的句子，则使用公式计算的方法可以计算出第 21 位的 Embedding。  
* 可以让模型容易地计算出相对位置，对于固定长度的间距 k，PE(pos+k) 可以用 PE(pos) 计算得到。因为 Sin(A+B) = Sin(A)Cos(B) + Cos(A)Sin(B), Cos(A+B) = Cos(A)Cos(B) - Sin(A)Sin(B)。  
将单词的词 Embedding 和位置 Embedding 相加，就可以得到单词的表示向量 x，x 就是 Transformer 的输入。  

## 3. 自注意力机制self-attention  
![self](https://pic4.zhimg.com/80/v2-f6380627207ff4d1e72addfafeaff0bb_720w.webp)  
上图是论文中 Transformer 的内部结构图，左侧为 Encoder block，右侧为 Decoder block。红色圈中的部分为 Multi-Head Attention，是由多个 Self-Attention组成的，可以看到 Encoder block 包含一个 Multi-Head Attention，而 Decoder block 包含两个 Multi-Head Attention (其中有一个用到 Masked)。Multi-Head Attention 上方还包括一个 Add & Norm 层，Add 表示残差连接 (Residual Connection) 用于防止网络退化，Norm 表示 Layer Normalization，用于对每一层的激活值进行归一化。  
self-attention的计算公式如下：  
$$Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V$$  
$$d_k是Q,K矩阵的维度$$   
在计算的时候需要用到矩阵Q(查询),K(键值),V(值)。在实际中，Self-Attention 接收的是输入(单词的表示向量x组成的矩阵X) 或者上一个 Encoder block 的输出。而Q,K,V正是通过 Self-Attention 的输入进行线性变换得到的。  
> 关于 $softmax(XX^T)X$   
> $X$ ：n个向量本身；  
> $XX^T$ ：向量与其他向量的乘积，表现为向量i对于每一个向量的相似度；  
> $softmax(XX^T)$ ：对每一行进行归一化处理，其使相似度归一化；  
> $softmax(XX^T)X$ ： 得到的结果词向量是经过加权求和之后的新表示，而权重矩阵是经过相似度和归一化计算得到的。  

## 4. 多头注意力机制multi-head attention  
Multi-Head Attention 包含多个 Self-Attention 层，首先将输入X分别传递到 h 个不同的 Self-Attention 中，计算得到 h 个输出矩阵Z。下图是 h=8 时候的情况，此时会得到 8 个输出矩阵Z。  
得到 8 个输出矩阵Z1到Z8之后，Multi-Head Attention 将它们拼接在一起 (Concat)，然后传入一个Linear层，得到 Multi-Head Attention 最终的输出Z。**Multi-Head Attention 输出的矩阵Z与其输入的矩阵X的维度是一样的。**  

对于每一个encoder块，输入 $X_(n×d)$ ，输出 $O_(n×d)$ ，其包含Multi-Head Attention, Add & Norm, Feed Forward, Add & Norm 组成。  
Add & Norm：残差连接+normalization：  
$$LayerNorm=(X+MultiHeadAttention(X))$$  
$$LayerNorm=(X+FeedForward(X))$$  

Feed Forward前馈网络是一个两层的全连接层，第一层的激活函数为 Relu，第二层不使用激活函数：  
$$X_out=max(0,XW_1+b_1)W_2+b_2$$  

对于每一个decoder块：
* 包含两个 Multi-Head Attention 层。  
* 第一个 Multi-Head Attention 层采用了 Masked 操作。  
> Masked:  
> 通过 Masked 操作可以防止第 i 个单词知道 i+1 个单词之后的信息。**注意 Mask 操作是在 Self-Attention 的 Softmax 之前使用的。即** $softmax(Masked(QK^T))V$     
![mask](https://pic2.zhimg.com/80/v2-35d1c8eae955f6f4b6b3605f7ef00ee1_720w.webp)   
* 第二个 Multi-Head Attention 层的K, V矩阵使用 Encoder 的编码信息矩阵C（即输出）进行计算，而Q使用上一个 Decoder block 的输出计算。  
* 最后有一个 Softmax 层计算下一个翻译单词的概率。  

## 5.参数计算  
先定：transformer有l个相同的层，每个层分为attention部分和MLP模块。  
attention：模型参数有 $Q,K,V$ ，对应权重矩阵和偏置 $W_Q,W_K,W_V$ 和偏置，输出权重矩阵 $W_o$ 和偏置,总参数为 $4h^2+4h$ 。     
MLP：两个线性层，分别从 $h\rightarrow4h$ 和 $4h\rightarrow h$ ,总参数为 $8h^2+5h$ 。  
所以，对于一个有l层的transformer来说，其参数量可近似表达为： $12lh^2+Vh$ 。   

# BERT  
**双向Transformer**  
以往的预训练模型的结构会受到单向语言模型（从左到右或者从右到左）的限制，因而也限制了模型的表征能力，使其只能获取单方向的上下文信息。而BERT利用MLM进行预训练并且采用深层的双向Transformer组件（单向的Transformer一般被称为Transformer decoder，其每一个token（符号）只会attend到目前往左的token。而双向的Transformer则被称为Transformer encoder，其每一个token会attend到所有的token。）来构建整个模型，因此最终生成能融合左右上下文信息的深层双向语言表征。  
**Bert的主体结构是由多个Transformer的堆叠。**  
BERT的输入为每一个token对应的表征（图中的粉红色块就是token，黄色块就是token对应的表征），并且单词字典是采用WordPiece算法来进行构建的。为了完成具体的分类任务，除了单词的token之外，作者还在输入的每一个序列开头都插入特定的分类token（[CLS]），该分类token对应的最后一个Transformer层输出被用来起到聚集整个序列表征信息的作用。  
由于BERT是一个预训练模型，其必须要适应各种各样的自然语言任务，因此模型所输入的序列必须有能力包含一句话（文本情感分类，序列标注任务）或者两句话以上（文本摘要，自然语言推断，问答任务）。那么如何令模型有能力去分辨哪个范围是属于句子A，哪个范围是属于句子B呢？BERT采用了两种方法去解决：  

1）在序列tokens中把分割token（[SEP]）插入到每个句子后，以分开不同的句子tokens。  

2）为每一个token表征都添加一个可学习的分割embedding来指示其属于句子A还是句子B。  

因此最后模型的输入序列tokens为下图（如果输入序列只包含一个句子的话，则没有[SEP]及之后的token）：  
![bertinput](https://pic1.zhimg.com/80/v2-a12ee6f717cc8312c43d140eb173def8_720w.webp)   
上面提到了BERT的输入为每一个token对应的表征，实际上该表征是由三部分组成的，分别是对应的token（词嵌入表示），分割（属于哪个句子的信息）和位置（整个序列中的位置信息） embeddings。与Transformer的位置嵌入不同的是，bert的嵌入表示都是可学习的。   
![biaozheng](https://pic1.zhimg.com/80/v2-ee823df66560850baa34128af76a6334_720w.webp)    
介绍完BERT的输入，实际上BERT的输出也就呼之欲出了，因为Transformer的特点就是有多少个输入就有多少个对应的输出，如下图：  
![out](https://pic3.zhimg.com/80/v2-7e0666db23ec2c29358cc89e2f823a06_720w.webp)   
C为分类token（[CLS]）对应最后一个Transformer的输出，Ti则代表其他token对应最后一个Transformer的输出。对于一些token级别的任务（如，序列标注和问答任务），就把Ti输入到额外的输出层中进行预测。对于一些句子级别的任务（如，自然语言推断和情感分类任务），就把C输入到额外的输出层中，这里也就解释了为什么要在每一个token序列前都要插入特定的分类。   
**BERT 可以用来干什么？**  
BERT 可以用于问答系统，情感分析，垃圾邮件过滤，命名实体识别，文档聚类等任务中，作为这些任务的基础设施即语言模型。BERT最重要的是实现了NLP领域的一个通用模型。  
