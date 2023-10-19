# BERT-基于Transformer encoder

**双向Transformer，只使用Transformer的encoder block**
以往的预训练模型的结构会受到单向语言模型（从左到右或者从右到左）的限制，因而也限制了模型的表征能力，使其只能获取单方向的上下文信息。

核心预训练任务：1.为了解决双向模型shotcut的问题，bert采用mask操作让模型来“单词填空”，避免了信息泄露。同时，为了避免模型过度集中在masked word上，mask操作还有分80%mask，10%nochange，10%random word。2.上下句预测。

而BERT利用MLM进行预训练并且采用深层的双向Transformer组件（单向的Transformer一般被称为Transformer decoder，其每一个token（符号）只会attend到目前往左的token。而双向的Transformer则被称为Transformer encoder，其每一个token会attend到所有的token。）来构建整个模型，因此最终生成能融合左右上下文信息的深层双向语言表征。
**Bert的主体结构是由多个Transformer encoder layer的堆叠。**

 ![1697727527573](image/Bert/1697727527573.png)
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
