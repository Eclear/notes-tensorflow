# minst进阶
>本篇学习构建一个TensorFlow模型的基本步骤，并将通过这些步骤为MNIST构建一个深度卷积神经网络。
## 0.mnist数据
使用官方文档中的脚本来自动下载和导入MNIST数据集。它会自动创建一个'MNIST_data'的目录来存储数据。
```
import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
```  
这里，mnist是一个轻量级的类。它以Numpy数组的形式存储着训练、校验和测试数据集。同时提供了一个函数，用于在迭代中获得minibatch，后面我们将会用到。  
## 1.运行TensorFlow的InteractiveSession
Tensorflow依赖于一个高效的C++后端来进行计算。与后端的这个连接叫做session。一般而言，使用TensorFlow程序的流程是先创建一个图，然后在session中启动它。  
  
这里，我们使用更加方便的InteractiveSession类。通过它，你可以更加灵活地构建你的代码。它能让你在运行图的时候，插入一些计算图，这些计算图是由某些操作(operations)构成的。这对于工作在交互式环境中的人们来说非常便利，比如使用IPython。如果你没有使用InteractiveSession，那么你需要在启动session之前构建整个计算图，然后启动该计算图。
## 2.计算图
为了在Python中进行高效的数值计算，我们通常会使用像NumPy一类的库，将一些诸如矩阵乘法的耗时操作在Python环境的外部来计算，每一个操作切换回Python环境时仍需要不小的开销，这一开销主要可能是用来进行数据迁移。  
  
TensorFlow也是在Python外部完成其主要工作，但是进行了改进以避免这种开销。其并没有采用在Python外部独立运行某个耗时操作的方式，而是先让我们描述一个交互操作图，然后完全将其运行在Python外部。这与Theano或Torch的做法类似。  
* ython代码的目的是用来构建这个可以在外部运行的计算图，以及安排计算图的哪一部分应该被运行。
## 3.构建Softmax 回归模型
>在这一节中我们将建立一个拥有一个线性层的softmax回归模型。在下一节，我们会将其扩展为一个拥有多层卷积网络的softmax回归模型。  

* 占位符：通过为输入图像和目标输出类别创建节点，来开始构建计算图。  
  >对于输入的数据，使用占位符
  ```
  x = tf.placeholder("float", shape=[None, 784])
  y_ = tf.placeholder("float", shape=[None, 10])
  ```
  输入图片x是一个2维的浮点数张量。这里，分配给它的shape为[None, 784]，其中784是一张展平的MNIST图片的维度。None表示其值大小不定，在这里作为第一个维度   值，用以指代batch的大小，意即x的数量不定。输出类别值y_也是一个2维张量，其中每一行为一个10维的one-hot向量,用于代表对应某一MNIST图片的类别。  
  
  虽然placeholder的shape参数是可选的，但有了它，TensorFlow能够自动捕捉因数据维度不一致导致的错误。  
  >None表数量不定，784和10表示数据的维数。  
  
* 变量  
  一个变量代表着TensorFlow计算图中的一个值，能够在计算过程中使用，甚至进行修改。在机器学习的应用过程中，模型参数一般用Variable来表示。
  ```
  W = tf.Variable(tf.zeros([784,10]))
  b = tf.Variable(tf.zeros([10]))
  ```  
  我们在调用tf.Variable的时候传入初始值。在这个例子里，我们把W和b都初始化为零向量。W是一个784x10的矩阵（因为我们有784个特征和10个输出值）。b是一个   10维的向量（因为我们有10个分类）。
