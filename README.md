# 摘要
> 机器学习的实现需要使用软件代码，从这一角度来说，机器学习也是软件工程的一部分。针对软件就会引入对软件是否正确的检验，那么在机器学习中如何进行软件测试？[相关源码](https://github.com/gdyshi/ml_test/blob/master/python/bugs.py)


---
# 什么是软件测试
> TDD-[测试驱动开发](https://baike.baidu.com/item/%E6%B5%8B%E8%AF%95%E9%A9%B1%E5%8A%A8%E5%BC%80%E5%8F%91/3328831?fr=aladdin)
是指开发功能代码之前，先编写测试代码，然后只编写使测试通过的功能代码，从而以测试来驱动整个开发过程的进行。这有助于编写简洁可用和高质量的代码，有很高的灵活性和健壮性，能快速响应变化，并加速开发过程。
安照测试覆盖面分白盒黑盒
安照测试阶段分单元集成和集成测试

# 为什么要进行软件测试
- 编写代码时因为个人习惯、手误或分心等产生的问题
> 如===写成==，for循环边界问题
- 对编程语言、调用库的特性不能足够深入的了解
> 矩阵的例子
```
import numpy as np
arr = np.ones(12)
print(arr)
print(arr.transpose())
```
- 对算法本身没有足够深入的了解
> 梯度的例子

# 机器学习中软件测试的特点
> 总体来看，大部分机器学习模型本质上就是低稳定性和高随机性的。主要原因在于数据计算部分（精度、溢出、计算本身的稳定性）
一般来说很难对整个测试集保证其正确或者错误，所以一般机器学习的测试集是用来测试根据训练集得到的模型是否在测试集上运行良好，既符合期望。

# 如何进行机器学习软件测试
## 白盒测试
- tensorflow测试
## 黑盒测试
一般机器学习的测试集是用来测试根据训练集得到的模型是否在测试集上运行良好，既符合期望。
# 机器学习中的测试重点
## 代码逻辑验证
> 特别是自己实现的条件和分支较多的代码。如tensorflow代码测试

## 算法验证
> 具体算法具体分析。基本原则是通过另一条途径，而非原始代码的实现路径来进行验证。

# 结论——如何防止机器学习编码中产生bug
## 编码模块最小化
> 这是软件工程中对代码的要求。分模块，各个模块尽量单一，简单。尽可能的将稳定性的代码从总体非稳定性代码中分离出来

## 尽量使用现有成熟库和方法，而不要自己写代码
> 现有成熟库和方法针对专项功能进行了深度优化，远比重新造轮子要快得多，也稳定得多。这样在编码效率、算法开销上都会提升

## 进行单元测试
> 使用现有机器学习框架所提供的测试功能，如TensorFlow的test_util.TensorFlowTestCase类

## 针对具体算法做相应的算法验证
- [梯度检验](http://ufldl.stanford.edu/wiki/index.php/%E6%A2%AF%E5%BA%A6%E6%A3%80%E9%AA%8C%E4%B8%8E%E9%AB%98%E7%BA%A7%E4%BC%98%E5%8C%96) [代码实现](https://github.com/gdyshi/ml_test/blob/master/python/grand.py)
> 在代码中使用导数公式来实现反向传播，验证时就根据极限法则来验证导数公式的正确性

---
参考资料
- [斯坦福教程-梯度检验](http://ufldl.stanford.edu/wiki/index.php/%E6%A2%AF%E5%BA%A6%E6%A3%80%E9%AA%8C%E4%B8%8E%E9%AB%98%E7%BA%A7%E4%BC%98%E5%8C%96)
- [UFLDL教程](http://ufldl.stanford.edu/wiki/index.php/UFLDL%E6%95%99%E7%A8%8B)
- [梯度检验](http://blog.csdn.net/tina_ttl/article/details/51034790)
-  [计算梯度的三种方法： 数值法，解析法，反向传播法](http://blog.csdn.net/raby_gyl/article/details/54407669)
-  [大规模Tensorflow网络的一些技巧](http://brightliao.me/2017/01/16/dl-workshop-massive-network-tips/)
-  [ Testing guide.](https://www.tensorflow.org/api_guides/python/test)
- [测试驱动开发](https://baike.baidu.com/item/%E6%B5%8B%E8%AF%95%E9%A9%B1%E5%8A%A8%E5%BC%80%E5%8F%91/3328831?fr=aladdin)