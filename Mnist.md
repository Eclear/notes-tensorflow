# mnist入门
## 1.数据集处理
* mnist 图片 -> 784维向量  
mnist 标签 -> 10维向量
## 2.Softmax回归（softmax regression）
* 用来分配不同对象的概率
* softmax回归分两步：
* 第一步：  
为了得到一张给定图片属于某个特定数字类的证据（softmax regression），我们对图片像素值进行加权求和。如果这个像素具有很强的证据说明这张图片不属于该类，那么相应的权值为负数，相反如果这个像素拥有有利的证据支持这张图片属于这个类，那么权值是正数。    
需要加入一个额外的偏置量（bias），因为输入往往会带有一些无关的干扰量。因此对于给定的输入图片 x 它代表的是数字 i 的证据可以表示为  
![t](http://www.tensorfly.cn/tfdoc/images/mnist1.png)  
其中wi代表权重, bi代表数字 i 类的偏置量，j 代表给定图片 x 的像素索引用于像素求和。然后用softmax函数可以把这些证据转换成概率 y：  
![t](http://www.tensorfly.cn/tfdoc/images/mnist4.png)  
softmax可以看成是一个激励（activation）函数或者链接（link）函数，把我们定义的线性函数的输出转换成我们想要的格式，也就是关于10个数字类的概率分布。  
![t](http://www.tensorfly.cn/tfdoc/images/mnist5.png)  
把输入值当成幂指数求值，再正则化这些结果值。这个幂运算表示，更大的证据对应更大的假设模型（hypothesis）里面的乘数权重值。反之，拥有更少的证据意味着在假设模型里面拥有更小的乘数系数。假设模型里的权值不可以是0值或者负值。Softmax然后会正则化这些权重值，使它们的总和等于1，以此构造一个有效的概率分布。  
对于softmax回归模型可以用下面的图解释，对于输入的xs加权求和，再分别加上一个偏置量，最后再输入到softmax函数中：  
![sr](http://www.tensorfly.cn/tfdoc/images/softmax-regression-scalargraph.png)  
把它写成一个等式，可以得到：  
![q](http://www.tensorfly.cn/tfdoc/images/softmax-regression-scalarequation.png)  
用向量表示这个计算过程：用矩阵乘法和向量相加。这有助于提高计算效率。（也是一种更有效的思考方式）  
![vector](http://www.tensorfly.cn/tfdoc/images/softmax-regression-vectorequation.png)  
更进一步，可以写成更加紧凑的方式：  
![7](http://www.tensorfly.cn/tfdoc/images/mnist7.png)
## 3.实现回归模型  
* TensorFlow把复杂的计算放在python之外完成，但是为了避免传输数据的开销，它做了进一步完善。Tensorflow不单独地运行单一的复杂计算，而是让我们可以先用图描述一系列可交互的计算操作，然后全部一起在Python之外运行。（这样类似的运行方式，可以在不少的机器学习库中看到。）  
使用TensorFlow之前，首先导入它：  
'import tensorflow as tf'  
过操作符号变量来描述这些可交互的操作单元，可以用下面的方式创建一个：  
'x = tf.placeholder("float", [None, 784])'  

