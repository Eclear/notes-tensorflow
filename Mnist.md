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
使用TensorFlow之前，首先导入它：  
`import tensorflow as tf`  
过操作符号变量来描述这些可交互的操作单元，可以用下面的方式创建一个：  
`x = tf.placeholder("float", [None, 784])`  
x不是一个特定的值，而是一个占位符placeholder，我们在TensorFlow运行计算时输入这个值。我们希望能够输入任意数量的MNIST图像，每一张图展平成784维的向量。我们用2维的浮点数张量来表示这些图，这个张量的形状是[None，784 ]。（这里的None表示此张量的第一个维度可以是任何长度的。）  
模型也需要权重值和偏置量，当然我们可以把它们当做是另外的输入（使用占位符），但TensorFlow有一个更好的方法来表示它们：Variable 。 一个Variable代表一个可修改的张量，存在在TensorFlow的用于描述交互性操作的图中。它们可以用于计算输入值，也可以在计算中被修改。  
```
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
```
赋予tf.Variable不同的初值来创建不同的Variable：在这里，我们都用全为零的张量来初始化W和b。因为我们要学习W和b的值，它们的初值可以随意设置。  
> W的维度是[784，10]，因为我们想要用784维的图片向量乘以它以得到一个10维的证据值向量，每一位对应不同数字类。b的形状是[10]，所以我们可以直接把它加到输出上面。  

现在，我们可以实现我们的模型啦。只需要一行代码！  
`y = tf.nn.softmax(tf.matmul(x,W) + b)`  
首先，我们用tf.matmul(X，W)表示x乘以W，对应之前等式里面的Wx，这里x是一个2维张量拥有多个输入。然后再加上b，把和输入到tf.nn.softmax函数里面。
> 机器学习从一个矩阵乘法开始！

* TensorFlow把复杂的计算放在python之外完成，但是为了避免传输数据的开销，它做了进一步完善。Tensorflow不单独地运行单一的复杂计算，而是让我们可以先用图描述一系列可交互的计算操作，然后全部一起在Python之外运行。（这样类似的运行方式，可以在不少的机器学习库中看到。） 
* TensorFlow不仅仅可以使softmax回归模型计算变得特别简单，它也用这种非常灵活的方式来描述其他各种数值计算，从机器学习模型对物理学模拟仿真模型。  
* 模型就可以在不同的设备上运行：计算机的CPU，GPU，甚至是手机！
## 4.训练模型  
在机器学习，我们通常定义指标来表示一个模型是坏的，这个指标称为成本（cost）或损失（loss），然后尽量最小化这个指标。  
一个非常常见的，非常漂亮的成本函数是“交叉熵”（cross-entropy）。交叉熵产生于信息论里面的信息压缩编码技术，但是它后来演变成为从博弈论到机器学习等其他领域里的重要技术手段。它的定义如下：  
![10](http://www.tensorfly.cn/tfdoc/images/mnist10.png)  
> y 是我们预测的概率分布, y' 是实际的分布（我们输入的one-hot vector)。比较粗糙的理解是，交叉熵是用来衡量我们的预测用于描述真相的低效性。

为了计算交叉熵，我们首先需要添加一个新的占位符用于输入正确值：  
`y_ = tf.placeholder("float", [None,10])`
