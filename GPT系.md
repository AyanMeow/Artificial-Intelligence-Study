# GPT

## GPT1

基于Transformer decoder（**去掉了没有mask的attention块**）

![1697722207606](image/GPT/1697722207606.png)

预训练技术：GPT-1使用了一种称为“生成式预训练”（Generative Pre-Training，GPT）的技术。预训练分为两个阶段：预训练和微调（fine-tuning）。在预训练阶段，GPT-1使用了大量的无标注文本数据集，例如维基百科和网页文本等。通过最大化预训练数据集上的log-likelihood来训练模型参数。在微调阶段，GPT-1将预训练模型的参数用于特定的自然语言处理任务，如文本分类和问答系统等。

GPT-1使用了BooksCorpus数据集[7]，这个数据集包含 7,000 本没有发布的书籍。作者选这个数据集的原因有二：1. 数据集拥有更长的上下文依赖关系，使得模型能学得更长期的依赖关系；2. 这些书籍因为没有发布，所以很难在下游数据集上见到，更能验证模型的泛化能力。

多层模型：GPT-1模型由12个堆叠的Transformer编码器组成，每个编码器包含多个注意力头和前向神经网络。这使得模型可以从多个抽象层次对文本进行建模，从而更好地捕捉文本的语义信息。这里的 Transformer 模块是经过变体后的结构，只包含 Decoder 中的Mask Multi-Head Attention 以及后面的 Feed Forward

### 无监督训练：

预训练过程的优化目标是最大化下面这个似然函数。

![L_1(U)=\sum_{i}\log P(u_i|u_{i-k},...,u_{i-1};\Theta )](https://latex.csdn.net/eq?L_1%28U%29%3D%5Csum_%7Bi%7D%5Clog%20P%28u_i%7Cu_%7Bi-k%7D%2C...%2Cu_%7Bi-1%7D%3B%5CTheta%20%29)

这个U就是一个 **tokens序列** ，其实就相当于一个文本，每个ui就是一个词。

k是 **窗口大小** ，也就是我们在预测某一个词的时候，关注到的它前面的序列的长度。

θ是这个 **模型的参数** 。

式子里的这个概率P是给出ui前面的k个词和模型参数θ的基础上，预测ui这个词出现的概率，因为用的是对数概率加和的形式，就相当于一个联合分布。这一部分的目标就是使用随机梯度下降法对模型参数进行更新，来使这个联合分布最大化。

**这里的无监督训练方式可以理解为直接让gpt照着无标注文本进行生成，从大量文本里学会哪些词该跟在哪些词后面。（最大化似然函数的意义）**

### 下游有监督微调

任务输入：

![1714311028241](image/GPT系/1714311028241.png)

## GPT2（实现multi-task）

GPT-2主要解决的问题是如何利用大规模未标注的自然语言文本来预训练一个通用的语言模型，从而提高自然语言处理的能力。

构造统一格式的数据集。把要解决的问题当做condition加在LLM输入里，这样语言模型就从 p(output|input)变成了 p(output|task_info, input) 。

> 例如，翻译任务的数据变成了（translate to french,  **english text** ,  *french text* ），QA任务的数据变成了（answer the question, document,  **question** ,  *answer* ）。

模型结构两者是差不多的，GPT2增大了模型规模，提出了117M、345M、762M、1542M四种不同规模的模型。同时，增加了vocabulary size，token数目增加到了50257。

以下是GPT-2的主要技术特点(其实除了规模大一点，和GPT-1变化不大)：

    模预训练：GPT-2使用了一种无监督学习的方法，在大规模文本语料库上进行预训练。在这个阶段，模型从语料库中学习文本序列的统计规律和语义信息。

    非监督多任务学习：GPT-2具有多任务学习的能力，通过训练模型来执行多个不同的自然语言处理任务，从而提高模型的鲁棒性和泛化能力。

> **语言建模（Language Modeling）** ：在语言建模任务中，模型接收一个文本序列作为输入，然后被要求预测序列中每个位置上的下一个词是什么。这个任务帮助模型学会理解文本序列中词语之间的语义和语法关系，从而提高了模型的文本生成能力。

> **掩码语言建模（Masked Language Modeling）** ：在掩码语言建模任务中，模型接收一个文本序列，其中部分词语会被随机掩码（即被替换为特殊的标记）。模型的目标是预测被掩码的词语是什么。这个任务帮助模型学会更好地理解和预测文本序列中缺失的部分，从而提高了对长文本的理解能力。

    Transformer架构：GPT-2使用Transformer架构作为模型的基础，使得模型可以自适应地处理长距离依赖关系，从而更好地理解文本的语义。

    无需人工标注数据：GPT-2在训练过程中不需要人工标注数据，可以自动从大规模文本语料库中学习自然语言的规律。

* **GPT2的位置编码是可学习的。**

零样本学习：GPT-2具有零样本学习的能力，能够在只看到少量样本的情况下学习和执行新任务。

## GPT3

GPT-3主要聚焦于更通用的NLP模型，解决当前BERT类模型的两个缺点：

1. **对领域内有标签数据的过分依赖** ：虽然有了预训练+精调的两段式框架，但还是少不了一定量的领域标注数据，否则很难取得不错的效果，而标注数据的成本又是很高的。
2. **对于领域数据分布的过拟合** ：在精调阶段，因为领域数据有限，模型只能拟合训练数据分布，如果数据较少的话就可能造成过拟合，致使模型的泛化能力下降，更加无法应用到其他领域。

因此GPT-3的主要目标是 **用更少的领域数据、且不经过精调步骤去解决问题** 。

传统方法：Fine-Tuning

GPT-3使用方法：

 **Few-Shot（FS）：** 指的是在推理时对模型进行一些任务相关的示例演示，**但不允许权重更新**。如图2.1所示，对于一个典型的数据集，一个示例具有上下文和所需的补全（例如英语句子和对应的法语句子），并通过给出K个示例上下文和补全的例子进行了Few-Shot。我们通常将K设置在10到100的范围内。FS的主要优点是，大大减少了对特定任务数据的需求，并减少了过拟合的可能性。主要缺点是，到目前为止，这种方法的结果要比最新的微调模型差很多。而且，仍然需要少量的任务特定数据。

 **One-Shot(1S)：** 和FS一样，不允许权重更新，但是k设置为1，和人类处理任务最为相似。

 **Zero-Shot (0S) ：** 没有示例演示，仅向模型提供描述任务的自然语言指令，同样没有权重更新。

GPT-3依旧延续自己的单向语言模型训练方式，只不过这次把模型尺寸增大到了 ***1750亿，*** 并且使用*45TB*数据进行训练。同时设置了各种size的模型进行对比。

# LLAMA

### LLAMA1架构

继最近对大型语言模型的研究之后，论文网络基于Transformer架构（Vaswani et al.，2017）。论文利用了随后提出的各种改进，并在不同的模型中使用，如 **PaLM** 。以下是与原始建筑的主要区别，以及论文在那里找到了这一变化的灵感：

 **预归一化[GPT3]。** 为了提高训练稳定性，对每个Transformer子层的输入进行归一化，而不是对输出进行归一化。使用了Zhang和Sennrich（2019）引入的**RMSNorm**规范化函数。

![1700984274070](image/GPT系/1700984274070.png)

 **SwiGLU激活功能[PaLM]。** 用Shazeer（2020）引入的**SwiGLU激活函数**取代了 **ReLU非线性** ，以提高性能。论文使用 $\frac{2}{3}4d$的尺寸，而不是PaLM中的4d。

 **旋转嵌入[GPTNeo]。** 删除了绝对位置嵌入，而是在网络的每一层添加了Su等人（2021）引入的 **旋转位置嵌入（RoPE）** 。

论文的模型使用**AdamW优化器。**

使用**因果多头注意力**的有效实现来减少内存使用和运行时间。

> 因果多头注意力算子是一种缩放点积注意力（SDPA）算子的变体，它常用于transformer模型。它计算了一个查询和一组键值对之间的注意力权重，其中键和值来自输入序列中的前面的词。这确保了注意力是因果的，意味着它不依赖于在推理时未知的未来词。
> 因果多头注意力算子是通过将查询、键和值张量分割成多个头，对每个头应用一个线性投影，计算每个头的SDPA，连接结果并应用另一个线性投影来实现的。这使得模型能够为每个头学习注意力函数的不同方面。
> 因果多头注意力算子是大型语言模型（LLM）的重要组成部分，这些模型可以从文本指令或少量示例执行新任务，例如GPT-3或LLaMA。为了提高训练效率，一些LLM使用了因果多头注意力算子的高效实现，减少了内存使用和计算时间 。

### llama2架构

Llama 2 采用了 Llama 1 的大部分预训练设置和模型架构。他们使用标准的Transformer架构，应用RMSNorm进行预归一化，使用SwiGLU激活函数和旋转位置编码。与 Llama 1 相比，主要的架构差异包括增加的上下文长度和分组查询注意力（GQA）。

> 这是一个新的注意力机制，可以提高大模型的推理可扩展性。它的工作原理是将键和值投影在多个头之间共享，而不会大幅降低性能。可以使用具有单个KV投影的原始多查询格式（MQA）或具有8KV投影的分组查询注意力变体（GQA）。即，在不同的head之间共享一对key和value就是MQA(multi-query-attention)，共享一组，例如8个key和value对的话，就是GQA（Group Query Attention）

### 预训练任务

常识推理，闭卷问答，阅读理解，数学推理，代码生成，大规模多任务语言理解

### 指令微调

指令微调是指通过构建指令格式的实例，然后以**有监督**的方式对大语言模型进行微调。指令格式通常包含任务描述，一对输入输出以及示例（可选）。

例如，“请回答以下问题：中国的首都是哪个城市？”，回答：“中国的首都是北京”，其中“请回答以下问题：”是任务描述，“中国的首都是哪个城市？”是输入，“中国的首都是北京”是输出。

指令微调可以帮助LLM **拥有更好的推理能力** ， 从而展现出**泛化到未见过任务**的卓越能力。也就是说，就算微调的指令中没有设计相关的任务，大模型在新任务上的表现也会优于微调之前。

## llama3

Llama3 由 Meta 最新公布的自建 24K GPU 集群上训练，使用超过 15T 的数据令牌，训练数据集是 Llama 2 的 7 倍，包括 4 倍的代码数据。

在上下文方面， Llama 3支持 **8K 的上下文长度**，是 Llama 2 容量的两倍，极大地提高了处理多步骤任务的能力。同时，该模型特别强调在理解、代码生成和指令跟随等复杂任务上的改进性能。

为了提高 Llama 3 模型的推理效率，Meta 采用了高效的分词器和**分组查询注意力（GQA）**，以及在大量公开数据上的预训练。

Meta 开发了一系列数据过滤管道。这些管道包括使用启发式过滤器、NSFW 过滤器、语义重复数据删除方法和文本分类器来预测数据质量。

为了寻求在真实世界中的优化， Meta 开发了一个新的高质量的人类评估集。该评估集包含1,800个提示，涵盖12个关键用例：征求建议，头脑风暴，分类，封闭式问题回答，编码，创意写作，提取，居住在角色/人物，开放式问题回答，推理，重写和总结。
