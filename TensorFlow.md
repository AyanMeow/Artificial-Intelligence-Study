# TensorFlow学习   
## 常用：   
* 从np.array或list创建tensor：tf.convert_to_tensor(x,dtype=) *   
* 创建0或1tensor：tf.zero(shape=[2,2,2]) ； tf.ones(shape=[1,1,1]) *  
* 填充固定：tf.fill *   
* 正态分布填充：tf.random.normal([],mean=1,stddev=1) *  
* 均匀分布填充：tf.random.uniform([],minval=1,maxval=2) *  
* 切片：正序：a[:]，倒序：a[::] *  
* tf.gather：该接口的作用：就是抽取出params的第axis维度上在indices里面所有的index对应的元素。  *  
* tf.gather_nd：将params索引为indices指定形状的切片数组中(indices代表索引后的数组形状)。  *  
* 在某一维合并：tf.concat([a,b],axis=c)。 *  
* 在某一维合并出新维度：tf.stack([a,b],axis=0)。 *  
* 在某一维完全拆解：tf.unstack(c,axis=2)。 *  
* 在某一维自定义拆解:tf.split(c,axis=2,num_or_size_splits=[1,2,2])。 *  
* 张量限制：值域：tf.clip_by_value(a,min,max)；非负：tf.relu(a)； *  
* 梯度剪裁，将权重限制在一个合适的范围内：L2范数值：tf.clip_by_norm(a,C);tf.clip_by_global_norm()。 *
* 
### 前向传播（Forward Propagation）：就是将上一层的输出作为下一层的输入，并计算下一层的输出，一直到运算到输出层为止。  
### 反向传播（Back Propagation）：反向传播仅指用于计算梯度的方法。而另一种算法，例如随机梯度下降法，才是使用该梯度来进行学习。原则上反向传播可以计算任何函数的到导数。  
反向传播算法的核心是代价函数 $C$ 对网络中参数（各层的权重 $W$ 和偏置 $b$ ）的偏导表达式 $\frac{\partial C}{\partial W}$ 和 $\frac{\partial C}{\partial b}$ 。这些表达式描述了代价函数值 $C$ 随权重 $W$ 或偏置 $b$ 变化而变化的程度。BP算法的简单理解：如果当前代价函数值距离预期值较远，那么我们通过调整权重 $W$ 或偏置 $b$ 的值使新的代价函数值更接近预期值（和预期值相差越大，则权重  $W$ 或偏置 $b$ 调整的幅度就越大）。一直重复该过程，直到最终的代价函数值在误差范围内，则算法停止。  
参考链接：https://zhuanlan.zhihu.com/p/447113449  
#### 简单来说，通过loss_func的值来更新前面每一层的  $W$ 和 $b$ 。  
### Broadcasting：TensorFlow在进行tensor的加减操作时，对于形状不同的tensor会自动补齐。   
