# FM算法相关


FM模型公式的如下

$$
y = w_0+\sum_{i=1}^{n}{w_ix_i}+\sum_{i=1}^{n}\sum_{j=i+1}^{n}<v_i,v_j>x_ix_j
$$

其中二次项是1到n的所有特征两两组合，这一项可以变形如下：

$$
\sum_{i=1}^{n}\sum_{j=i+1}^{n}<v_i,v_j>x_ix_j=\frac{1}{2}\sum_{f=1}^{k}[(\sum_{i=1}^{n}v_{i,f}x_i)^2-\sum_{i=1}^{n}v_{i,f}^2x_i^2]
$$

## **Q：为什么要引入隐向量？**

A：为了用两个隐向量的內积模拟二次项的参数，从而极大降低参数个数，并且缓解二次项稀疏的问题。

假设有1万个特征，特征的两两组合，那么二次项就有 𝐶100002=49995000这么权重。

而加入隐向量后，可以看公式2的等号右边：中括号内只有N的复杂度，中括外边是k的复杂度，因此总的复杂度就降到了 𝑘𝑁 。考虑到k是隐向量的维度，可以根据效果和复杂度来人为指定，是远远小于特征数的，比如取k为16，则 𝑘𝑁=16∗10000=160000 。
