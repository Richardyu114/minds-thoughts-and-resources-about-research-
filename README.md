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

## *2019.1.8*

### *<u>1.分布式机器学习</u>*

分布式机器学习的目的是为了解决目前训练数据大，计算资源有限，同时训练过程人为设定和死板的问题。。。*<u>这个分布式机器学习似乎与网络的去中心化思想有点类似</u>*，这个对于我们计算机视觉，三维建模有什么帮助呢？或许就是能够实时地认清周围的环境和世界。。

https://mp.weixin.qq.com/s/yI075kYz3y1Lq51o19wjNA

另有一书，《分布式机器学习：理论，算法与实践 》



### *<u>2. Best paper awards in CS(since 1996)</u>*

https://jeffhuang.com/best_paper_awards.html



### *<u>3.最近一年semantic SLAM的代表工作</u>*

语义 SLAM 的难点在于怎样设计误差函数，将 Deep Learning 的检测或者分割结果作为一个观测，融入 SLAM 的优化问题中一起联合优化，同时还要尽可能做到至少 GPU 实时。

就仅针对视觉而言，有ETH的SVO，ICRA2017的最佳论文Probabilistic Data Association for Semantic SLAM等，大多的工作都比较初步，大多数看到的还算是利用网络来融合语义标签，地图的构建是基于几何约束构建的，那么是否可以利用新的学习方法直接进行语义地图的构建，同时尽量使得地图的边缘清晰。




### <u>*4.用latex beamer做学术汇报PPT*</u>

https://www.overleaf.com/learn/latex/Beamer_Presentations:_A_Tutorial_for_Beginners_(Part_1)%E2%80%94Getting_Started

http://www.latexstudio.net/archives/2825.html

### *<u>5.在没有导师的帮助下，如何寻找并做好自己的科研</u>*

https://www.zhihu.com/question/23647187?utm_source=wechat_session&utm_medium=social&utm_oi=706533037407481856

以我自己的经历来看，最重要的还是自己对自己研究方向的把控，这一点特别需要有专业的人给与指导和启发，同时引发自己的思考，然后在调研中确立。在研究学习的过程中，如果能有几个志同道合的人一起交流，一起质疑，对idea的产生和修正都大有裨益。此外一些科研人员必备的素养和技能，这都是需要自己在平时中养成良好的习惯，然后一点一滴积累的。当然，目前的技术发展已经如此迅速，没有办法从头开始，毕竟时间不长，因此需要我们有针对性地看待问题，站在巨人的肩膀上，先从别人那里获得经验和加成，先模仿然后尽可能快速地进入自己的研究和创新。


## *2019.1.9*

### *<u>1.computer vision research groups</u>*

http://www.cs.cmu.edu/~cil/v-groups.html

## *2019.1.10*

### *<u>1.一个博客网站“超立方体”</u>*

博主的兴趣是几何视觉，还有pointNet，里面有一些项目，与自己相关，可以看看。

https://hypercube.top/



### *<u>2.语义分割和实例分割</u>*

何恺明的论文Panoptic Feature Pyramid Networks以及dynamic SLAM都用到了该网络

何恺明的主站：http://kaiminghe.com/ 

R-CNN-->FAST R-CNN-->FASTER R-CNN-->MASK R-CNN

目标检测，实例分割，语义分割

一篇博文进行概述：https://towardsdatascience.com/deep-learning-for-object-detection-a-comprehensive-review-73930816d8d9


总体的趋向是从单一的架构到复杂的，集成的网络化发展。上述网络的初衷应该是完成场景中的目标检测，比如可以用在无人驾驶中的行人检测。对于视觉slam来说，一是可以进行场景识别，提高回环检测的速度和效率，另一个是建图方面，对于建好的图进行实例分割和语义分割，但是问题是，什么样的图能被该网络利用，点云地图？还有，这样建图的意义何在，不能光想着建图，而没有一个落脚点，否则就是跟风和技术的堆砌。

### *<u>3.工作应该落在何处</u>*

完整的visual slam框架很大，里面涉及许多内容，大体是定位和建图，中间会穿插着些优化，以提高精度，减小运行体量。目前基于学习的方法引入了视觉slam，主要是集中在训练数据，达到自动生成位姿和深度图，意义不大，似乎并没有看到基于学习的建图方法。

关键词虽然是语义地图，动态地图，稠密精细地图，但是目的和意义是什么，难道就是娱乐？。。。。


### *<u>4.和姚师兄交流</u>*

![image](https://github.com/Richardyu114/-/blob/master/images/the%20position%20of%203d%20reconstruction.png)


3D在无人驾驶中的定位，虽然上述系统比较低级，但是高级的东西在于感知和路径规划方面，这一点大多现在倾向用基于学习的方法。。。用于避障的精度地图。。。？


## *2019.1.11*

### *<u>1.Top 50 matplotlib Visualizations</u>*

https://www.machinelearningplus.com/plots/top-50-matplotlib-visualizations-the-master-plots-python

### 

### *<u>2.在PyTorch中用单个2D图像创建3D模型</u>*

https://chenhsuanlin.bitbucket.io/3D-point-cloud-generation/


## *2019.1.13*



### *<u>1.2018 年度 GtiHub 开源项目 TOP 25：数据科学 & 机器学习</u>*

https://mp.weixin.qq.com/s/tpB003Ow6HkA6J1a_gh_ww

特别关注 fast-ai, pytorch, Facebook 的 Detectron 


## *2019.1.15*

### *<u>1.MIT deep learning</u>*

https://deeplearning.mit.edu/


### <u>*2.李飞飞实验室等最近的语义分割新进展*</u>

[auto-deeplab](https://arxiv.org/pdf/1901.02985v1.pdf)





## *2019.1.17*


### *<u>1.6D目标姿态估计，李飞飞夫妇等提出[DenseFusion](https://arxiv.org/pdf/1901.04780.pdf)</u>*


对 RGB-D 输入的已知物体进行 6D 姿态估计。该方法的核心是在每个像素级别嵌入、融合 RGB 值和点云，这和之前使用图像块计算全局特征 [41] 或 2D 边界框 [22] 的研究相反。这种像素级融合方法使得本文的模型能够明确地推理局部外观和几何信息，这对处理重度遮挡情况至关重要。此外，研究者还提出了一种迭代方法，能够在端到端学习框架中完成姿态微调。这极大地提高了模型性能，同时保证了实时推理速度

想法：是否可以结合上次看到的一篇论文，语义分割和目标检测结合的，然后再利用这个deepfusion来进行一次组合，目的是是为了让机器得到更多的有关周围环境的信息，从而进行下一步的机器人行动。


## *2019.1.19*

### *<u>1. ETH苏黎世联邦理工大学的[AnyRobotics](https://github.com/ANYbotics)公司</u>*


ANYbotics 团队推出了一项激动人心的解决方案：ANYmal。这是一款非常复杂的四足机器人，专为解决恶劣工业环境中的挑战而设计。它的体积和中等大小的狗差不多，体重为 30kg。它结合了高端计算机系统和稳健的硬件。为了在不同的恶劣环境中自主作业，它配备了传感系统来执行搜索和援救行动、检查和其它监视任务。它的软件似乎比波士顿动力公司演示的机器人更具适应性。

从[机器之心宣传的内容来看](https://mp.weixin.qq.com/s/ajlaieiFOO2wx4sQPykGag)，其似乎运用了视觉SLAM的知识，体现在了建图以及路劲规划上，物体检测与识别可能运用了机器学习或者open CV库，目前还不清楚，有待进一步调研。这项工程可以被用作自己毕业设计的和背景甚至是研究课题的大环境。
  
 
 ### *<u>2. Facebook的深度学习3D重建新进展：[deepSDF](https://arxiv.org/pdf/1901.05103.pdf)</u>*
 
扭曲、空洞、体素化仍然是很多 3D重建模型的通病，导致视觉效果很不友好。Facebook、MIT 等近期提出了新型的基于深度学习的连续场 3D 重建模型 DeepSDF，可以生成具备复杂拓扑的高质量连续表面。特别是，由于可以在反向传播过程中顺便求得空间梯度，DeepSDF 能生成非常平滑的表面。

该研究的贡献包括：使用连续隐式表面进行生成式形状 3D 建模；基于概率自解码器的 3D 形状学习方法；展示了该方法在形状建模和补全上的应用。该模型可生成具备复杂拓扑的高质量连续表面，并在形状重建和补全方面的量化对比中获得了当前最优结果。举例来说，该模型仅使用 7.4 MB 的内存来表示形状的完整类别（如数千个 3D 椅子模型），这比单个未压缩 512^3 3D 位图内存占用（16.8 MB）的一半还要少。

### *<u>3. 一篇关于无人驾驶现状的[综述](https://arxiv.org/ftp/arxiv/papers/1901/1901.04407.pdf)。可用于寻找毕业设计的背景。</u>*


### *<u>4. [Now anyone can train Imagenet in 18 minutes](https://www.fast.ai/2018/08/10/fastai-diu-imagenet/)</u>*


### *<u>5. IBM的新研究：[intuition learning](https://goodboyanush.github.io/resources/intuition_learning.pdf)</u>*

直觉学习这个东西期盼已久，毕竟深度学习对于数据的依赖和转移性导致了它不能完成多任务。因此类似与迁移学习，如何在已知的经验下，去估计未知的东西。。这也许对视觉SLAM的场景识别与理解会有帮助。

## *2019.1.24*
### *<u>1. 瑞士哥德堡大学的Artificial Neural Networks课程[FFR135](http://physics.gu.se/~frtbm/joomla/index.php?option=com_content&view=article&id=124&Itemid=509)</u>*

近日，哥德堡大学物理系 Bernhard Mehlig 教授在 arXiv 上发布了他的一本「新书」[《Artifical Neural Networks》](https://arxiv.org/pdf/1901.05639.pdf)。这本书正是他根据在哥德堡大学物理系 2018 秋季学期教学（FFR315）过程中的笔记整理而成。在这门课程中，他结合物理学（特别是统计物理学）的知识详细讲述了机器学习中神经网络在物理学中的各种应用，包括深度学习、卷积网络、强化学习，以及其他各种有监督和无监督机器学习算法。

Hopfield 网络是一种可以识别或重构图像的人工神经网络，它通过某种方法（Hebb 规则）分配权重，并将图像储存在人工神经网络中。这是一种非常经典的想法，它构成了玻尔兹曼机和深度信念网络等方法的基础，但目前深度神经网络能代替它完成模式识别任务。


作者表示课程将 Hopfield 网络作为第一部分主要有三个原因，首先很多后续的深度神经网络都基于相同的构建块，以及与 Hebb 规则相近的学习方法。其次 Hopfield 网络可以很好地解决最优化问题，且最终算法与马尔可夫链蒙特卡洛方法密切相关，这在物理及统计学上非常重要。最后，Hopfield 网络与物理的随机系统密切相关，可能这也是最重要的吧。

不过书中默认读者是了解物理知识的人，落脚点也在物理，因此选择性地读一读。

### *<u>2. GAN用于视觉里程计--[GANVO](https://arxiv.org/pdf/1809.05786.pdf)</u>*

以前的工作大多是基于监督学习，无监督学习来进行视觉里程计的估计，输出位姿和深度图。不过平心而论，其鲁棒性和精度和ORB-SLAM2相比还存在差距。正如此论文中所说的“有监督的深度学习方法广泛应用于视觉里程计领域，但是在没有丰富标签数据的环境中是不可行的。另一方面，在VO研究中，对在未知环境中使用无标签数据进行定位和建图的无监督深度学习方法相较而言获得较少关注度”。

在该论文中，作者提出一种生成式无监督学习框架，通过使用深度卷积生成式对抗神经网络从无标签RGB影像序列中预测包含6自由度的相机运动姿态以及单目深度图。作者通过变换视图序列以及最小化在多视姿态生成与单视图深度生成网络中所使用到的目标损失函数来产生一种监督信号。其主要贡献有：
- 本文首次提出在单目VO中使用对抗式神经网络和循环无监督学习方法来联合估计姿态和深度图；

- 作者提出一种新的对抗式技术，使得GANs可以在没有深度真值信息的情况下生成深度影像；

- 相较于传统VO方法，本文算法在姿态和深度估计中没有严格精确的参数调整过程

## *2019.1.25*

### *<u>1. [Sam Roweis](https://cs.nyu.edu/~roweis/)的[machine learning](https://cs.nyu.edu/~roweis/csc2515/)</u>*
Sam Roweis是个非常有想法的学者，这门课的PPT写的很不错，看完CS229之后没事拿出来翻一翻。

此外，[Neural Network for Machine Learning by Geoffrey Hinton](https://www.youtube.com/watch?v=cbeTc-Urqak&list=PLoRl3Ht4JOcdU872GhiYWf6jwrk_SNhz9)也不错，Hinton老爷子毕生研究神经网络，对这个东西理解的非常深刻。。
[相关笔记和作业](https://github.com/liufuyang/course-Neural-Networks-for-Machine-Learning)


## *2019.1.30*

### *<u>1. [一个基于单目的三位视觉重建综述](https://mp.weixin.qq.com/s/ihrExTygb-Pnnh4o4tAYnQ)，里面简单地介绍了SFM和深度学习重建的原理和缺陷。需要后期进行总结。</u>*

重建流程：
多视角图像-->图像特征提取匹配-->稀疏重建SfM-->稠密重建MVS-->点云模型化-->三维模型

### *<u>2. [移动端单人姿态估计](https://github.com/edvardHua/PoseEstimationForMobile)</u>*

估计的是人的骨架变化，并不是常规的长距离定位。


### *<u>2.CSDN博客上对SLAM系统的论文和相关环节进行了梳理和材料推荐的[汇总帖](https://blog.csdn.net/heyijia0327/article/details/82855443)，大多还是基于VIO及一些数学基础，可以看看。</u>*
  
## *2019.2.2*

### *<u>1. [史上引用次数最多的机器学习论文 Top 10](https://mp.weixin.qq.com/s/fiVyhyQwnOR-AXkj5Y_Ztg)</u>*
- EM回归
- logistic回归
- 随机森林
- 分类与回归数
- 支持向量机开源库 libsvm
- 统计学习理论
- 主成分分析
- 决策树
- 深度卷积神经网络
- 支持向量机

## *2019.2.5*

### *<u>1. [使用Python和Mask R-CNN自动寻找停车位](https://mp.weixin.qq.com/s/USkdhyEAiU-ZGqmw0v3fXQ)</u>*
这是机器之心公众号转载的medium上的文章，是通过python爬数据和Mask R-CNN进行语义分割然后向用户发送车库信息。


## *2019.2.13*

### *<u>1. 有关深度学习-计算机视觉的极限和瓶颈问题</u>*

[这个论文](https://arxiv.org/pdf/1805.04025.pdf)说明了现在深度学习加速下的计算机视觉子啊语义分割，分类等信息提取与认知问题上的一些固有问题：

- 总是需要大量标注数据做训练；
- 数据集表现良好，真是场景往往不具有适应性；
- 对于图像的改变过于敏感，比如视角的改变，或者添加一些“小细节”就可以欺骗网络；

这些问题引来了组合爆炸的问题，就是物体是有限的，但是之间的组合，导致背景，物体和图片层次之间的多样性。

因此，需要涉及组合原则和因果模型的互补方法，以捕捉数据的基本结构。此外，面对组合性爆炸，我们要再次思考如何训练和评估视觉算法。


## *2019.2.15*

### *<u>1. [2018年度10大突破性计算机视觉文章](https://www.topbots.com/most-important-ai-computer-vision-research/#ai-cv-paper-2018-4)</u>*

- Spherical CNNs        :open_mouth:
- Adversarial Examples that Fool both Computer Vision and Time-Limited Humans
- A Closed-form Solution to Photorealistic Image Stylization
- Group Normalization
- Taskonomy: Disentangling Task Transfer Learning              :open_mouth:
- Self-Attention Generative Adversarial Networks
- GANimation: Anatomically-aware Facial Animation from a Single Image
- Video-to-Video Synthesis                :open_mouth:
- Everybody Dance Now
- Large Scale GAN Training for High Fidelity Natural Image Synthesis
  
  
 ### *<u>2. Face++ detection组最近在做的语义分割工作，其中也包含了对语义分割的[梳理](https://mp.weixin.qq.com/s/nB4FAKss1A5jmgKG3hdRoA)</u>*
  
 
 ## *2019.2.21*
 
 ### *<u>1. [人工智能子领域高被引用学者名单](https://www.aminer.cn/mostinfluentialscholar)</u>*
 
| [Theory](https://www.aminer.cn/mostinfluentialscholar/theory) | ACM Symposium on Theory of Computing<br/>IEEE Annual Symposium on Foundations of Computer Science |
| ------------------------------------------------------------ | :----------------------------------------------------------- |
| [Artificial Intelligence](https://www.aminer.cn/mostinfluentialscholar/ai) | AAAI Conference on Artificial Intelligence<br/>International Joint Conference on Artificial Intelligence |
| [Machine Learning](https://www.aminer.cn/mostinfluentialscholar/ml) | Annual Conference on Neural Information Processing Systems<br/>International Conference on Machine Learning |
| [Data Mining](https://www.aminer.cn/mostinfluentialscholar/datamining) | ACM SIGKDD International Conference on Knowledge Discovery and Data Mining |
| [Database](https://www.aminer.cn/mostinfluentialscholar/database) | ACM SIGMOD International Conference on Management of Data<br/>International Conference on Very Large Data Bases |
| [Multimedia](https://www.aminer.cn/mostinfluentialscholar/mm) | ACM International Conference on Multimedia                   |
| [Security](https://www.aminer.cn/mostinfluentialscholar/security) | ACM Conference on Computer and Communications Security <br/>IEEE Symposium on Security and Privacy |
| [System](https://www.aminer.cn/mostinfluentialscholar/system) | ACM Symposium on Operating Systems Principles<br/>USENIX Symposium on Operating Systems Design and Implementation |
| [Software Engineering](https://www.aminer.cn/mostinfluentialscholar/software) | International Conference on Software Engineering             |
| [Computer Networking](https://www.aminer.cn/mostinfluentialscholar/networking) | ACM Conference on Special Interest Group on Data Communication |
| [Natural Language Processing](https://www.aminer.cn/mostinfluentialscholar/nlp) | Annual Meeting of the Association for Computational Linguistics |
| [Human-Computer Interaction](https://www.aminer.cn/mostinfluentialscholar/hci) | ACM CHI Conference on Human Factors in Computing Systems     |
| [Computer Graphics](https://www.aminer.cn/mostinfluentialscholar/graphics) | International Conference and Exhibition on Computer Graphics and Interactive Techniques |
| [Computer Vision](https://www.aminer.cn/mostinfluentialscholar/cv) | IEEE Conference on Computer Vision and Pattern Recognition<br/>IEEE International Conference on Computer Vision |
| [Web and Information Retrieval](https://www.aminer.cn/mostinfluentialscholar/webir) | International World Wide Web Conference<br/>International ACM SIGIR Conference on Research and Development in Information Retrieval |

  
  
  
