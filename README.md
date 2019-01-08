# 个人笔记
这是私人项目，记录平时一些随便看到的有关科研和学习的东西



## ***2018.12.4:***



### <u>*1. Adversarial Robustness - Theory and Practice*</u>

Neur**IPS 2018** 入选论文，目测应该是与**GAN**有关



<https://adversarial-ml-tutorial.org/>



### *<u>2.机器学习在传感器网络中的运用</u>*

现在固定的场景，场景里面预定好标签，对应于使用的传感器，先跑一边让网络学习，训练后在以后跑的时候自动识别标签然后定位



## ***2018.12.5:***



### *<u>1.机器学习理论中用到的数学知识点：</u>*

| 算法或理论     |                          数学知识点                          |
| :------------- | :----------------------------------------------------------: |
| 贝叶斯分类器   | 随机变量，贝叶斯公式，随机变量独立性，正态分布，最大似然估计 |
| 决策树         |                      概率，熵，Gini洗漱                      |
| KNN算法        |                           距离函数                           |
| 主成分分析     |    协方差矩阵，散布矩阵，拉格朗日乘数法，特征值与特征向量    |
| 流形学习       |     流形，最优化，测地线，测地距离，图，特征值与特征向量     |
| 线性判别分析   |      散度矩阵，逆矩阵，拉格朗日乘数法，特征值与特征向量      |
| 支持向量机     | 点到平面的距离，Slater条件，强对偶，拉格朗日对偶，KKT条件，凸优化，核函数，Mercer条件 |
| logistic       |   概率，随机变量，最大似然估计，梯度下降法，凸优化，牛顿法   |
| 随机森林       |                          抽样，方差                          |
| AdaBoost算法   |          概率，随机变量，极值定理，数学期望，牛顿法          |
| 隐马尔可夫模型 | 概率，离散随机变量，条件概率，随机变量独立性，拉格朗日乘数法，最大似然估计 |
| 条件随机场     |               条件概率，数学期望，最大似然估计               |
| 高斯混合模型   |             正态分布，最大似然估计，Jensen不等式             |
| 人工神经网络   |                     梯度下降法，链式法则                     |
| 卷积神经网络   |                     梯度下降法，链式法则                     |
| 循环神经网络   |                     梯度下降法，链式法则                     |
| 生成对抗网络   | 梯度下降法，链式法则，极值定理，Kullback-Leibler散度，Jensen-Shannon散度，测地距离，条件分布，互信息 |
| k-means算法    |                           距离函数                           |
| 贝叶斯网络     |                   条件概率，贝叶斯公式，图                   |
| VC维           |                       Hoeffding不等式                        |

“我觉得，高数基础，看看多元函数微分学，线性回归。概率统计里面重点看看，全概率公式，贝叶斯定理，几大分布，中心极限定理，参数估计，假设检验。线代看看矩阵理论，特征向量，二次型。视频可以直接找张宇李永乐，王式安，把那几张内容讲解的视频抽出来看看就好了。这样子应该最快，不用专门去一门课一门课的复习。需要哪块知识，去看那块就行了。



我觉得现阶段推导过程，有所了解，知道怎么来的就行了吧。**重点理解算法思想，解决什么问题，为什么提出来。重点在实践，把这些基本算法都自己实现一下，走一下这个流程，做一些小型的项目**。



把机器学习基本算法实现了，再说深度学习。做做，手写识别系统这种小东西就好吧。他们研究生上课做的实验，就和机器学习实战这书里面的小项目差不多。”



### *<u>2.PCA主成分分析—统计信息降维</u>*

PCA解释：<https://www.zhihu.com/question/41120789> 

<https://blog.csdn.net/lyl771857509/article/details/79435402> 

协方差矩阵：<https://www.zhihu.com/question/36348219/answer/275378672> 

独立成分分析与主成分分析：<https://www.zhihu.com/question/28845451/answer/42292804> 



## ***2018.12.6:***



### *<u>1.线性代数两本：</u>*

introduction to linear algebra —Gilbert Strang 先看

linear algebra and its application —David C. Lay 后看

<http://www.math.ucla.edu/~tao/resource/general/115a.3.02f/> 陶哲轩的讲义也可辅助



## ***2018.12.16.   2018.12.23***



### *<u>1.逻辑回归</u>*

过拟合：学习了噪声，或者特征过多

L1 L2正则化，修正过拟合，加速梯度下降

L1适合少数特征，它能自动选择特征，

l2适和多种特征



### *<u>2.机器学习中的凸函数</u>*

两点中值函数值小于两点函数值中值

**凸优化问题的局部极小值是全局极小值**，利用反证法证明，矛盾局部极小值的定义

Hessian矩阵半正定等价于函数为凸函数

凸函数需要在凸集上面讨论才有意义

Jessen不等式—f(E(x))<=E(f(x)). f是凸函数  ----- EM算法



### *<u>3.拉格朗日乘子法    KKT</u>*

无约束条件，等式约束条件

拉格朗日乘数法是一种寻找多元函数在其变量受到一个或多个条件的约束时的极值的方法



KKT是原问题转换为对偶问题的必要条件，但不是充分条件，只有在求解一个凸优化问题时，才是充分必要条件

**瑞典皇家理工学院（KTH）“统计学习基础”课程**

<http://www.csc.kth.se/utbildning/kth/kurser/DD3364/Schedule.php> 



## ***2019.1.5***

### *<u>1.人工智能和数据科学的七大python库</u>*

<https://mp.weixin.qq.com/s/EK7ZQW8a7TohisEDUvPEaA>



### *<u>2.伊利诺伊大学厄巴纳香槟分校(University of Illinois, Urbana-Champaign)的计算机科学教授Jeff Erickson公开了即将出版的免费电子教科书《算法》。</u>*

Jeff Erickson，计算机科学教授，加州大学伯克利分校计算机科学博士毕业，1998年起就职于伊利诺伊大学厄巴那香槟分校(University of Illinois, Urbana-Champaign)，研究兴趣领域为算法和数据结构等，主要教授大型算法课程，根据其个人主页信息，他的课堂讲义大受学生欢迎。

<https://mp.weixin.qq.com/s/GvNwgqOChiMJpxH4NUuiLA>



### *<u>3.美国加州大学伯克利分校教授、机器人与强化学习领域专家 Pieter Abbeel 发布了一份资源大礼：《深度学习与机器人学》105页PPT。</u>*

这份PPT整理自Abbeel教授2018年受邀参加的69个演讲，内容涵盖监督学习、强化学习和无监督学习的重要进展，以及深度学习的主要应用等方面，有助于读者对深度学习和机器人学有一个宏观的理解。

<https://www.dropbox.com/s/dw4kmxkrv3orujd/2018_12_xx_Abbeel--AI.pdf?dl=0>



### *<u>4.从2018年1月到12月，Mybridge网站比较了近2.2万篇机器学习文章，考虑文章受欢迎程度、参与度、新近度等因素，评估其质量，从中精选出Top 50榜单</u>*

<https://mp.weixin.qq.com/s/bHnTNhlugEpTCXUJewb7sg>



### *<u>5.伯克利《人工智能导论》(2018)课程，介绍智能计算机系统设计的基本思想和技术。重点内容是统计和决策理论建模范式。本文课程中所学习到的技术将适用于各种各样的人工智能问题</u>*

<https://mp.weixin.qq.com/s/CpsqTBEMiXqgXCIAA9VA6w>



### *<u>6.可视化统计学概率论入门</u>*

<https://seeing-theory.brown.edu/cn.html>



### *<u>7.哈佛大学生物统计学和流行病学教授 Miguel Hernan 和 Jamie Robins 最近一直在写一本书，并希望为因果推理的概念和方法提供一个连贯性的介绍。</u>*

他们表示目前因果推理相关的材料大部分都分散在多个学科的期刊上，或者存在各种技术文章中。他们希望这本书能帮助各种对因果推断感兴趣的读者，不论是流行病学家、统计学家、社会科学学家还是计算机科学家。这本书主要分为三部分，它们的难度依次递增：不带模型的因果推理、带模型的因果推理、复杂长跨度数据的因果推理。

<https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/>



### *<u>8.MIT科学家Dimitri P. Bertsekas今日发布了一份2019即将出版的《强化学习与最优控制》书稿及讲义，该专著目的在于探索这人工智能与最优控制的共同边界，形成一个可以在任一领域具有背景的人员都可以访问的桥梁。</u>*

<https://mp.weixin.qq.com/s/wSck7B2GMQ_k3urqrrJICA>

## ***2019.1.6***

### *1.受欢迎的AI 网课*

https://hn.academy/

