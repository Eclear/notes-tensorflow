# mnist入门
## 1.数据集处理
* mnist 图片 -> 784维向量  
mnist 标签 -> 10维向量
## 2.Softmax回归（softmax regression）
* 用来分配不同对象的概率
* softmax回归分两步：
* 第一步：  
为了得到一张给定图片属于某个特定数字类的证据（softmax regression），我们对图片像素值进行加权求和。如果这个像素具有很强的证据说明这张图片不属于该类，那么相应的权值为负数，相反如果这个像素拥有有利的证据支持这张图片属于这个类，那么权值是正数。  
* 第二步：  
![t]（http://www.tensorfly.cn/tfdoc/images/mnist1.png）
