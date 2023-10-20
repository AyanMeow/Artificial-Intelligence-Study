# T5

**Transfer Text-to-Text Transformer**

**Transformer 的 Encoder-Decoder 模型** 

**将所有 NLP 任务都转化成 Text-to-Text （文本到文本）任务** 。

![1697781416563](image/T5/1697781416563.png)

\<Task:info\>的形式。将 NLP 任务都转换成 Text-to-Text 形式，也就可以 **用同样的模型，同样的损失函数，同样的训练过程，同样的解码过程来完成所有 NLP 任务** 。

**Data：C4。作者从 Common Crawl（一个公开的网页存档数据集，每个月大概抓取 20TB 文本数据） 里清出了 750 GB 的训练数据**

![1697781623284](image/T5/1697781623284.png)

总体架构：

* Transformer Encoder-Decoder 模型；
* BERT-style 式的破坏方法；
* Replace Span 的破坏策略；
* 15 %的破坏比；
* 3 的破坏时小段长度。

## 架构探索

首先作者们先对预训练模型中的多种模型架构（Transformer）进行了比对，最主要的模型架构可以分成下面三种。

![](https://pic2.zhimg.com/80/v2-b1a8d9af6110e6d1b6a7615fc300a229_720w.webp)

第一种， **Encoder-Decoder 型** ，即 Seq2Seq 常用模型，分成 Encoder 和 Decoder 两部分，对于 Encoder 部分，输入可以看到全体，之后结果输给 Decoder，而 Decoder 因为输出方式只能看到之前的。此架构代表是 MASS（今年WMT的胜者），而 BERT 可以看作是其中 Encoder 部分。

第二种， 相当于上面的  **Decoder 部分** ，当前时间步只能看到之前时间步信息。典型代表是 GPT2 还有最近 CTRL 这样的。

第三种， **Prefix LM（Language Model） 型** ，可看作是上面 Encoder 和 Decoder 的融合体，一部分如 Encoder 一样能看到全体信息，一部分如 Decoder 一样只能看到过去信息。最近开源的 UniLM 便是此结构。

**Text-to-Text 架构中，Encoder-Decoder 模型效果最好**

## **对预训练的探索：**

![1697781868342](image/T5/1697781868342.png)

第一个方面， **高层次方法（自监督的预训练方法）对比** ，总共三种方式。

1. **语言模型式** ，就是 GPT-2 那种方式，从左到右预测；
2. **BERT-style 式** ，就是像 BERT 一样将一部分给破坏掉，然后还原出来；
3. Deshuffling （顺序还原）式，就是将文本打乱，然后还原出来。

Bert-style 最好，进入下一轮。

第二方面，对文本一部分进行 **破坏时的策略** ，也分三种方法。

1. **Mask 法** ，如现在大多模型的做法，将被破坏 token 换成特殊符如 [M]；
2. **replace span（小段替换）法** ，可以把它当作是把上面 Mask 法中相邻 [M] 都合成了一个特殊符，每一小段替换一个特殊符，提高计算效率；
3. **Drop 法** ，没有替换操作，直接随机丢弃一些字符。

![1697782015950](image/T5/1697782015950.png)

此轮获胜的是 **Replace Span 法**

第三方面，到底该**对文本百分之多少进行破坏**呢，挑了 4 个值，10%，15%，25%，50%，最后发现 BERT 的 **15%** 就很 ok了。

第四方面，因为 Replace Span 需要决定 **对大概多长的小段进行破坏** ，于是对不同长度进行探索，2，3，5，10 这四个值，最后发现 **3** 结果最好。

# BART
