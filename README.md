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

## *2019.2.23*

### *<u>1. 新书：可解释机器学习，帮助读者更好的理解机器学习的内在模型。[Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/) </u>*
  

## *2019.2.24*

### *<u>1. 在medium上发现一个作者对计算机视觉任务发展现状的做的一个网络梳理</u>*
- **Image Classification**

[[LeNet](https://medium.com/@sh.tsang/paper-brief-review-of-lenet-1-lenet-4-lenet-5-boosted-lenet-4-image-classification-1f5f809dbf17)] [[AlexNet](https://medium.com/coinmonks/paper-review-of-alexnet-caffenet-winner-in-ilsvrc-2012-image-classification-b93598314160)] [[ZFNet](https://medium.com/coinmonks/paper-review-of-zfnet-the-winner-of-ilsvlc-2013-image-classification-d1a5a0c45103)] [[VGGNet](https://medium.com/coinmonks/paper-review-of-vggnet-1st-runner-up-of-ilsvlc-2014-image-classification-d02355543a11)] [[Highway](https://towardsdatascience.com/review-highway-networks-gating-function-to-highway-image-classification-5a33833797b5)] [[SPPNet](https://medium.com/coinmonks/review-sppnet-1st-runner-up-object-detection-2nd-runner-up-image-classification-in-ilsvrc-906da3753679)] [[PReLU-Net](https://medium.com/coinmonks/review-prelu-net-the-first-to-surpass-human-level-performance-in-ilsvrc-2015-image-f619dddd5617)] [[STN](https://towardsdatascience.com/review-stn-spatial-transformer-network-image-classification-d3cbd98a70aa)] [[DeepImage](https://medium.com/@sh.tsang/review-deep-image-a-big-data-solution-for-image-recognition-99e5f7b1c802)] [[GoogLeNet / Inception-v1](https://medium.com/coinmonks/paper-review-of-googlenet-inception-v1-winner-of-ilsvlc-2014-image-classification-c2b3565a64e7)] [[BN-Inception / Inception-v2](https://medium.com/@sh.tsang/review-batch-normalization-inception-v2-bn-inception-the-2nd-to-surpass-human-level-18e2d0f56651)] [[Inception-v3](https://medium.com/@sh.tsang/review-inception-v3-1st-runner-up-image-classification-in-ilsvrc-2015-17915421f77c)] [[Inception-v4](https://towardsdatascience.com/review-inception-v4-evolved-from-googlenet-merged-with-resnet-idea-image-classification-5e8c339d18bc)] [[Xception](https://towardsdatascience.com/review-xception-with-depthwise-separable-convolution-better-than-inception-v3-image-dc967dd42568)] [[MobileNetV1](https://towardsdatascience.com/review-mobilenetv1-depthwise-separable-convolution-light-weight-model-a382df364b69)] [[ResNet](https://towardsdatascience.com/review-resnet-winner-of-ilsvrc-2015-image-classification-localization-detection-e39402bfa5d8)] [[Pre-Activation ResNet](https://towardsdatascience.com/resnet-with-identity-mapping-over-1000-layers-reached-image-classification-bb50a42af03e)] [[RiR](https://medium.com/@sh.tsang/review-rir-resnet-in-resnet-image-classification-be4c79fde8ba)] [[RoR](https://towardsdatascience.com/review-ror-resnet-of-resnet-multilevel-resnet-image-classification-cd3b0fcc19bb)] [[Stochastic Depth](https://towardsdatascience.com/review-stochastic-depth-image-classification-a4e225807f4a)] [[WRN](https://towardsdatascience.com/review-wrns-wide-residual-networks-image-classification-d3feb3fb2004)] [[FractalNet](https://medium.com/datadriveninvestor/review-fractalnet-image-classification-c5bdd855a090)] [[Trimps-Soushen](https://towardsdatascience.com/review-trimps-soushen-winner-in-ilsvrc-2016-image-classification-dfbc423111dd)] [[PolyNet](https://towardsdatascience.com/review-polynet-2nd-runner-up-in-ilsvrc-2016-image-classification-8a1a941ce9ea)] [[ResNeXt](https://towardsdatascience.com/review-resnext-1st-runner-up-of-ilsvrc-2016-image-classification-15d7f17b42ac)] [[DenseNet](https://towardsdatascience.com/review-densenet-image-classification-b6631a8ef803)] [[PyramidNet](https://medium.com/@sh.tsang/review-pyramidnet-deep-pyramidal-residual-networks-image-classification-85a87b60ae78)]

- **Object Detection**

[[OverFeat](https://medium.com/coinmonks/review-of-overfeat-winner-of-ilsvrc-2013-localization-task-object-detection-a6f8b9044754)] [[R-CNN](https://medium.com/coinmonks/review-r-cnn-object-detection-b476aba290d1)] [[Fast R-CNN](https://medium.com/coinmonks/review-fast-r-cnn-object-detection-a82e172e87ba)] [[Faster R-CNN](https://towardsdatascience.com/review-faster-r-cnn-object-detection-f5685cb30202)] [[DeepID-Net](https://towardsdatascience.com/review-deepid-net-def-pooling-layer-object-detection-f72486f1a0f6)] [[R-FCN](https://towardsdatascience.com/review-r-fcn-positive-sensitive-score-maps-object-detection-91cd2389345c)] [[ION](https://towardsdatascience.com/review-ion-inside-outside-net-2nd-runner-up-in-2015-coco-detection-object-detection-da19993f4766)] [[MultiPathNet](https://towardsdatascience.com/review-multipath-mpn-1st-runner-up-in-2015-coco-detection-segmentation-object-detection-ea9741e7c413)] [[NoC](https://medium.com/datadriveninvestor/review-noc-winner-in-2015-coco-ilsvrc-detection-object-detection-d5cc84e372a)] [[G-RMI](https://towardsdatascience.com/review-g-rmi-winner-in-2016-coco-detection-object-detection-af3f2eaf87e4)] [[TDM](https://medium.com/datadriveninvestor/review-tdm-top-down-modulation-object-detection-3f0efe9e0151)] [[SSD](https://towardsdatascience.com/review-ssd-single-shot-detector-object-detection-851a94607d11)] [[DSSD](https://towardsdatascience.com/review-dssd-deconvolutional-single-shot-detector-object-detection-d4821a2bbeb5)] [[YOLOv1](https://towardsdatascience.com/yolov1-you-only-look-once-object-detection-e1f3ffec8a89)] [[YOLOv2 / YOLO9000](https://towardsdatascience.com/review-yolov2-yolo9000-you-only-look-once-object-detection-7883d2b02a65)] [[YOLOv3](https://towardsdatascience.com/review-yolov3-you-only-look-once-object-detection-eab75d7a1ba6)] [[FPN](https://towardsdatascience.com/review-fpn-feature-pyramid-network-object-detection-262fc7482610)] [[RetinaNet](https://towardsdatascience.com/review-retinanet-focal-loss-object-detection-38fba6afabe4)] [[DCN](https://towardsdatascience.com/review-dcn-deformable-convolutional-networks-2nd-runner-up-in-2017-coco-detection-object-14e488efce44)]

- **Semantic Segmentation**

[[FCN](https://towardsdatascience.com/review-fcn-semantic-segmentation-eb8c9b50d2d1)] [[DeconvNet](https://towardsdatascience.com/review-deconvnet-unpooling-layer-semantic-segmentation-55cf8a6e380e)] [[DeepLabv1 & DeepLabv2](https://towardsdatascience.com/review-deeplabv1-deeplabv2-atrous-convolution-semantic-segmentation-b51c5fbde92d)] [[SegNet](https://towardsdatascience.com/review-segnet-semantic-segmentation-e66f2e30fb96)] [[ParseNet](https://medium.com/datadriveninvestor/review-parsenet-looking-wider-to-see-better-semantic-segmentation-aa6b6a380990)] [[DilatedNet](https://towardsdatascience.com/review-dilated-convolution-semantic-segmentation-9d5a5bd768f5)] [[PSPNet](https://towardsdatascience.com/review-pspnet-winner-in-ilsvrc-2016-semantic-segmentation-scene-parsing-e089e5df177d)] [[DeepLabv3](https://towardsdatascience.com/review-deeplabv3-atrous-convolution-semantic-segmentation-6d818bfd1d74)]

- **Biomedical Image Segmentation**

[[CUMedVision1](https://medium.com/datadriveninvestor/review-cumedvision1-fully-convolutional-network-biomedical-image-segmentation-5434280d6e6)] [[CUMedVision2 / DCAN](https://medium.com/datadriveninvestor/review-cumedvision2-dcan-winner-of-2015-miccai-gland-segmentation-challenge-contest-biomedical-878b5a443560)] [[U-Net](https://towardsdatascience.com/review-u-net-biomedical-image-segmentation-d02bf06ca760)] [[CFS-FCN](https://medium.com/datadriveninvestor/review-cfs-fcn-biomedical-image-segmentation-ae4c9c75bea6)] [[U-Net+ResNet](https://medium.com/datadriveninvestor/review-u-net-resnet-the-importance-of-long-short-skip-connections-biomedical-image-ccbf8061ff43)]

- **Instance Segmentation**

[[DeepMask](https://towardsdatascience.com/review-deepmask-instance-segmentation-30327a072339)] [[SharpMask](https://towardsdatascience.com/review-sharpmask-instance-segmentation-6509f7401a61)] [[MultiPathNet](https://towardsdatascience.com/review-multipath-mpn-1st-runner-up-in-2015-coco-detection-segmentation-object-detection-ea9741e7c413)] [[MNC](https://towardsdatascience.com/review-mnc-multi-task-network-cascade-winner-in-2015-coco-segmentation-instance-segmentation-42a9334e6a34)] [[InstanceFCN](https://towardsdatascience.com/review-instancefcn-instance-sensitive-score-maps-instance-segmentation-dbfe67d4ee92)] [[FCIS](https://towardsdatascience.com/review-fcis-winner-in-2016-coco-segmentation-instance-segmentation-ee2d61f465e2)]

- **Super Resolution**

[[SRCNN](https://medium.com/coinmonks/review-srcnn-super-resolution-3cb3a4f67a7c)] [[FSRCNN](https://towardsdatascience.com/review-fsrcnn-super-resolution-80ca2ee14da4)] [[VDSR](https://towardsdatascience.com/review-vdsr-super-resolution-f8050d49362f)] [[ESPCN](https://medium.com/datadriveninvestor/review-espcn-real-time-sr-super-resolution-8dceca249350)] [[RED-Net](https://medium.com/datadriveninvestor/review-red-net-residual-encoder-decoder-network-denoising-super-resolution-cb6364ae161e)] [[DRCN](https://medium.com/datadriveninvestor/review-drcn-deeply-recursive-convolutional-network-super-resolution-f0a380f79b20)] [[DRRN](https://towardsdatascience.com/review-drrn-deep-recursive-residual-network-super-resolution-dca4a35ce994)] [[LapSRN & MS-LapSRN](https://towardsdatascience.com/review-lapsrn-ms-lapsrn-laplacian-pyramid-super-resolution-network-super-resolution-c5fe2b65f5e8)]
 
 
 ## *2019.2.27*
 
 ### *<u>1.[GSLAM](https://github.com/zdzhaoyong/GSLAM)</u>*
 
 一个SLAM框架的执行和评价平台，用来测试不同SLAM解决方案的差异。
 
 ### *<u>2.2000-2018历年CVPR最佳论文[清单](https://mp.weixin.qq.com/s/4VpVguXgwE_Rj9m-vNpsIw)</u>*
 
 其中注意下:
 
 - [2018].[Taskonomy: Disentangling Task Transfer Learning](http://taskonomy.stanford.edu/taskonomy_CVPR2018.pdf)
 
论文研究了一个非常新颖的课题，那就是研究视觉任务之间的关系，根据得出的关系可以帮助在不同任务之间做迁移学习。该论文提出了「Taskonomy」——一种完全计算化的方法，可以量化计算大量任务之间的关系，从它们之间提出统一的结构，并把它作为迁移学习的模型。实验设置上，作者首先找来一组一共 26 个任务，当中包括了语义、 2D、2.5D、3D 任务，接着为任务列表里的这 26 个任务分别训练了 26 个任务专用神经网络。结果显示，这些迁移后的模型的表现已经和作为黄金标准的任务专用网络的表现差不多好。论文提供了一套计算和探测相关分类结构的工具，其中包括一个求解器，用户可以用它来为其用例设计有效的监督策略。

- [2016]. [Deep Residual Learning for Image Recognition](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)

在现有基础下，想要进一步训练更深层次的神经网络是非常困难的。我们提出了一种减轻网络训练负担的残差学习框架，这种网络比以前使用过的网络本质上层次更深。我们明确地将这层作为输入层相关的学习残差函数，而不是学习未知的函数。同时，我们提供了全面实验数据，这些数据证明残差网络更容易优化，并且可以从深度增加中大大提高精度。我们在 ImageNet 数据集用 152 层--比 VGG 网络深 8 倍的深度来评估残差网络，但它仍具有较低的复杂度。在 ImageNet 测试集中，这些残差网络整体达到了 3.57% 的误差。该结果在 2015 年大规模视觉识别挑战赛分类任务中赢得了第一。此外，我们还用了 100 到 1000 层深度分析了的 CIFAR-10。

- [2015]. [DynamicFusion: Reconstruction and Tracking of Non-rigid Scenes in Real-Time](https://rse-lab.cs.washington.edu/papers/dynamic-fusion-cvpr-2015.pdf)

作者提出第一个结合商用传感器对 RGBD 扫描结果进行捕获，该结果可实时重建非刚性变形场景的密集 SLAM 系统。被称作 DynamicFusion 的这种方法在重建场景几何的当儿，还能同时估算一个密集体积的 6D 运动场景，并将估算结果变成实时框架。与 KinectFusion 一样，该系统可以生成越来越多去噪、保留细节、结合多种测量的完整重建结果，并实时显示最新的模型。由于该方法无需基于任何模板或过往的场景模型，因此适用于大部分的移动物体和场景。

- [2014]. [What Object Motion Reveals About Shape With Unknown BRDF and Lighting](https://cseweb.ucsd.edu/~ravir/differentialtheory.pdf)

作者提出了一种理论，用于解决在未知远距离照明以及未知各向同性反射率下，运动物体的形状识别问题，无论是正交投影还是穿透投影。该理论对表面重建硬度增加了基本限制，与涉及的方法无关。在正交投影场景下，三个微分运动在不计 BRDF 和光照的情况下，可以产生一个将形状与图像导数联系起来的不变量。而在透视投影场景下，四个微分运动在面对未知的 BRDF 与光照情况，可以产生基于表面梯度的线性约束。此外，论文也介绍了通过不变量实现重建的拓扑类。

最后，论文推导出一种可以将形状恢复硬度与场景复杂性联系起来的通用分层。从定性角度来说，该不变量分别是用于简单照明的均匀偏微分方程，以及用于复杂照明的非均匀方程。从数量角度来说，该框架表明需要更多的最小运动次数来处理更复杂场景的形状识别问题。关于先前假设亮度恒定的工作，无论是 Lambertian BRDF 还是已知定向光源，一律被被当作是分层的特殊情况。作者利用合成与真实数据进一步说明了重建方法可以如何更好地利用这些框架。

- [2010] [Efficient Computation of Robust Low-Rank Matrix Approximations in the Presence of Missing Data using the L1 Norm](https://acvtech.files.wordpress.com/2010/06/robustl1_eriksson.pdf)

低秩近似矩阵计算是许多计算机视觉应用中的基础操作。这类问题的主力解决方案一直是奇异值分解（Singular Value Decomposition）。一旦存在数据缺失和异常值，该方法将不再适用，遗憾的是，我们经常在实践中遇到这种情况。

论文提出了一种计算矩阵的低秩分解法，一旦丢失数据时会主动最小化 L1 范数。该方法是 Wiberg 算法的代表——在 L2 规范下更具说服力的分解方法之一。通过利用线性程序的可区分性，可以对这种方法的基本思想进行扩展，进而包含 L1 问题。结果表明，现有的优化软件可以有效实现论文提出的算法。论文提供了令人信服、基于合成与现实数据的初步实验结果。

- [2009] [Single Image Haze Removal Using Dark Channel Prior](http://www.jiansun.org/papers/Dehaze_CVPR2009.pdf)

本文中提出了一个简单却有效、针对单个输入图像的暗通道去雾法。暗通道先验去雾法是一种户外去雾图像的统计方法，它主要基于一个关键的观察——室外无雾图像中的大多数局部斑块包含一些像素，这些像素的强度起码有一个颜色通道处于低状态。使用这种基于雾度成像模型的先验方法，我们可以直接估计图像的雾霾厚度，借此将图像恢复至高质量的无雾状态。各种模糊图像的去雾结果证明了论文所提出先验方法的成效。此外，我们可以通过该方法获得高质量的深度图。

- 后面的也有几篇与SfM以及三维重建有关，但是年代较为久远，可以通过其他最新综述获得。


## *2019.2.28*

### *<u>1. 深度学习机制中的[F-Principle](https://mp.weixin.qq.com/s?__biz=MzIwMTc4ODE0Mw==&mid=2247494694&idx=1&sn=7020fb834ce8307f27ce9c072047d37d&chksm=96ea33a6a19dbab0a6585daa00d5b5c65501dd633fa677c80541fad0e170d92baffe379315c3&scene=21#wechat_redirect) </u>*

深度学习倾向于优先使用低频来拟合目标函数。我们将这个机制称为 F-Principle（频率原则）。

[该论文](https://arxiv.org/pdf/1901.06523.pdf)中对这种现象的研究可以进一步帮助研究者建立函数拟合的收敛过程。


## *2019.3.1*

### *<u>1. MIT的[6.S191](http://introtodeeplearning.com/)  </u>*

理论与实践结合，可以作为项目做做。




## *2019.3.6*

### *<u>1. [基于SegNet和U-Net的遥感图像语义分割](https://www.cnblogs.com/skyfsm/p/8330882.html)</u>*
一位比较厉害的[博主](https://www.cnblogs.com/skyfsm/)参加比赛一个开源项目分享，也是语义分割，对自己刚开始学习很有帮助。另外这个博主的文章也多看看，学习学习。


## *2019.3.12*

### *<u>1. [saliency Maps](https://www.analyticsindiamag.com/what-are-saliency-maps-in-deep-learning/)显著图</u>*

The model considers three features in an image, namely colours, intensity and orientations. These combinations are presented in the saliency map. 


## *2019.3.14*

### *<u>1. 语义SLAM（semantic SLAM）梳理</u>*

- 语义信息和定位是相互扶助的，语义信息可以帮助定位，SLAM也能提高语义理解的水平。
真正的semantic SLAM，语义信息是要能够帮助定位的，比如这篇：Probabilistic data association for semantic slam (ICRA'17)。用object detection的结果作为SLAM前端的输入，跟ORB之类的特征互补提高定位鲁棒性。优点很明显，这下SLAM不会因为你把床收拾了一下就啥都不认识了（视觉特征都变了，但床还是床）。难点是detection结果的data association最好能跟定位联合优化，但前者是个离散问题。这篇文章用EM算法，E步考虑所有可能的association，比较粗暴，但识别物体较少的时候还不错（论文实验里只识别椅子）

- SLAM也能提升语义理解水平。前面提到的SemanticFusion和类似的工作里，融合了多个视角语义理解结果的3D地图，其中的语义标签准确率高于单帧图像得到的结果，这很容易理解。另外，通过在3D空间引入一些先验信息，比如用CRF对地图做一下diffusion，能进一步提升准确率。但CRF毕竟还是简单粗暴，如果设计更精细的滤波算法，尤其是能从真实数据中学习一些先验的话，应该效果还会更好。这方面的工作还没有。

- 融合优化之后的结果如果反馈给图像语义理解算法做一下fine-tuning，那就是self-supervised learning了。这方面的工作也还没有。

语义分割帮助SLAM提高定位精度，建立语义地图，也就是真正的semantic slam 。但是获取语义信息不只有语义分割这一种，还包括目标识别，物体检测和实例分割，所以下面我给出的论文不仅仅限于用语义分割获取语义信息。

 语义信息用于bundle adjustment
 
（1）Joint Detection, Tracking and Mapping by Semantic Bundle Adjustment

（2）Improving Constrained Bundle Adjustment Through Semantic Scene Labeling

（3）Semantic segmentation–aided visual odometry for urban autonomous driving

 将语义信息用到优化公式里面


 语义信息用于定位
 
 （1）Localization from semantic observations via the matrix permanent
 
 （2）Probabilistic Data Association for Semantic SLAM
 
 （3）Semantic Pose using Deep Networks Trained on Synthetic RGB-D
 
 （4）X-View: Graph-Based Semantic Multi-View Localization
 
 （5）Pop-up SLAM: Semantic Monocular Plane SLAM for Low-texture Environments
 
  场景理解用于改善状态估计，尤其是在低纹理区域，是目前极少的开源语义SLAM方案之一
  
  三个开源的语义SLAM方案
  
  (1)DA-RNN_Semantic Mapping with Data Associated [yuxng/DA-RNN](https://link.zhihu.com/?target=https%3A//github.com/yuxng/DA-RNN)
  
  (2)SemanticFusion: Dense 3D Semantic Mapping with Convolutional Neural Networks[dysonroboticslab / SemanticFusion - Bitbucket](https://link.zhihu.com/?target=https%3A//bitbucket.org/dysonroboticslab/semanticfusion/overview)
  
  (3) Pop-up SLAM: Semantic Monocular Plane SLAM[shichaoy/pop_up_image](https://github.com/shichaoy/pop_up_image)


### *<u>2. 高翔关于语义SLAM的一些[看法](https://mp.weixin.qq.com/s/ayL4TUSkqI57Sg_4xKlCoQ)</u>*

语义SLAM出现的动机的一大部分原因视觉地图的作用没发挥上来，大家都着眼于提高定位的精度，地图的构建，重利用上存在很大的问题。正如高翔所说的“语义SLAM的概念很模糊。你会找到许多带着『Semantic』字眼，实际上完全不在说同一件事情的论文。比如从**图像到Pose端到端的VO、从分割结果建标记点云、场景识别、CNN提特征、CNN做回环、带语义标记误差的BA**，等等，都可以叫语义SLAM。但是从实用层面考虑，我觉得最关键的一点是：**用神经网络帮助SLAM提路标**”...

“所以『把物体建出来当地图路标』其实是一个不错的思路。**剩下的就是看物体有多少类，能不能支持到大多数常见的物体**。就自动驾驶来说，有了车道线，你至少就能知道自己在第几根车道线之间。有了车道线地图，就能知道自己在地图上哪两根车道线之间。类别再丰富一些，能用来定位的东西就更多，覆盖范围也就更宽。这个算是语义SLAM和传统SLAM中最不同的地方了。”
 
 ### *<u>3. [技术刘](http://www.liuxiao.org/)总结的Semantic SLAM的几篇重点文章</u>*
 
 1. 《Probabilistic Data Association for Semantic SLAM》 ICRA 2017
 
语义 SLAM 中的概率数据融合，感觉应该算开山鼻祖的一篇了。这篇也获得了 ICRA 2017 年的 Best Paper，可见工作是比较早有创新性的。文章中引入了 EM 估计来把语义 SLAM 转换成概率问题，优化目标仍然是熟悉的重投影误差。这篇文章只用了 DPM 这种传统方法做检测没有用流行的深度学习的检测网络依然取得了一定的效果。当然其文章中有很多比较强的假设，比如物体的三维中心投影过来应该是接近检测网络的中心，这一假设实际中并不容易满足。不过依然不能掩盖其在数学上开创性的思想。

[文章](http://www.liuxiao.org/wp-content/uploads/2018/08/Probabilistic-Data-Association-for-Semantic-SLAM.pdf)

2. 《VSO: Visual Semantic Odometry》 ECCV 2018
 
既然检测可以融合，把分割结果融合当然是再自然不过的想法，而且直观看来分割有更加细粒度的对物体的划分对于 SLAM 这种需要精确几何约束的问题是更加合适的。ETH 的这篇文章紧随其后投到了今年的 ECCV 2018。这篇文章依然使用 EM 估计，在上一篇的基础上使用距离变换将分割结果的边缘作为约束，同时依然利用投影误差构造约束条件。在 ORB SLAM2 和 PhotoBundle 上做了验证取得了一定效果。这篇文章引入距离变换的思路比较直观，很多人可能都能想到，不过能够做 work 以及做了很多细节上的尝试，依然是非常不容易的。但仍然存在一个问题是，分割的边缘并不代表是物体几何上的边缘，不同的视角这一分割边缘也是不停变化的，因此这一假设也不是非常合理。

[文章](http://www.liuxiao.org/wp-content/uploads/2018/08/VSO-Visual-Semantic-Odometry.pdf)

3. 《Stereo Vision-based Semantic 3D Object and Ego-motion Tracking for Autonomous Driving》 ECCV 2018

港科大沈邵劼老师团队的最新文章，他们的 VINS 在 VIO 领域具有很不错的开创性成果。现在他们切入自动驾驶领域做了这篇双目语义3D物体跟踪的工作，效果还是很不错的。在沈老师看来，SLAM 是一个多传感器融合的框架，RGB、激光、语义、IMU、码盘等等都是不同的观测，所以只要是解决关于定位的问题，SLAM 的框架都是一样适用的。在这篇文章中，他们将不同物体看成不同的 Map，一边重建一边跟踪。使用的跟踪方法仍然是传统的 Local Feature，而 VIO 作为世界坐标系的运动估计。语义融合方面，他们构造了4个优化项：feature reprojection error, object size prior, motion residual, bounding box reprojection error，最终取得了不错的效果。

[文章](http://www.liuxiao.org/wp-content/uploads/2018/08/Stereo-Vision-based-Semantic-3D-Object-and-Ego-motion-Tracking-for-Autonomous-Driving.pdf)
[视频](https://www.youtube.com/watch?v=5_tXtanePdQ)


4. 《Long-term Visual Localization using Semantically Segmented Images》ICRA 2018

这篇论文讲得比较有意思，它不是一个完整的SLAM系统，不能解决Mapping的问题。它解决的问题是，当我已经有了一个很好的3D地图后，我用这个地图怎么来定位。在传统方法中，我们的定位也是基于特征匹配的，要么匹配 Local Feature 要么匹配线、边等等几何特征。而我们看人在定位时的思维，其实人看不到这么细节的特征的，通常人是从物体级别去定位，比如我的位置东边是某某大楼，西边有个学校，前边有个公交车，我自己在公交站牌的旁边这种方式。当你把你的位置这样描述出来的时候，如果我自己知道你说的这些东西在地图上的位置，我就可以基本确定你在什么地方了。这篇文章就有一点这种意思在里边，不过它用的观测结果是分割，用的定位方法是粒子滤波。它的地图是三维点云和点云上每个点的物体分类。利用这样语义级别的约束，它仍然达到了很好的定位效果。可想而知这样的方法有一定的优点，比如语义比局部特征稳定等；当然也有缺点，你的观测中的语义信息要比较丰富，如果场景中你只能偶尔分割出一两个物体，那是没有办法work的。

[文章](http://www.liuxiao.org/wp-content/uploads/2018/08/Long-term-Visual-Localization-using-Semantically-Segmented-Images.pdf)
[演示](https://www.youtube.com/watch?v=M55qTuoUPw0)
 
 
 ### *<u>3. 一些有关SLAM的work的slides</u>*
 
 1. RSS 2015 [Setting future goals and indicators of progress for SLAM](http://ylatif.github.io/movingsensors/)
 
 2. ICCV 2015 [The Future of Real-Time SLAM: 18th December 2015 (ICCV Workshop)](http://wp.doc.ic.ac.uk/thefutureofslam/programme/)
 
    [a related Blog](http://www.computervisionblog.com/2016/01/why-slam-matters-future-of-real-time.html)
    
 3. ICRA 2017 [First International Workshop on Event-based Vision](http://rpg.ifi.uzh.ch/ICRA17_event_vision_workshop.html)
 
 4. ECCV 2018 [Visual Localization Feature-based vs. Learned Approaches](https://sites.google.com/view/visual-localization-eccv-2018/home)
 


## *2019.3.20*

### *<u>1.一位在谷歌大脑工作的人博客[Free Mind](http://freemind.pluskid.org/)  </u>*
看他的简历是浙大竺可桢学院毕业的，之后再MIT CSAIL读的PhD,现在做的也是机器学习，算法，以及数学，脑神经等相关的，同时也看出热爱文学，绘画和艺术，值得学习借鉴。


## *2019.3.22*

### *<u>1. 计算机视觉比较有名的和自己感兴趣的教授 </u>*
[网站列表](http://peipa.essex.ac.uk/info/groups.html)

1.[stanford vision and learning lab](http://svl.stanford.edu/)

2.[Douglas Lanman](http://alumni.media.mit.edu/~dlanman/index.html)做结构光和虚拟现实技术等

3.[UCLA朱松纯](http://www.stat.ucla.edu/~sczhu/)

4.[Berkeley CV Group](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/) Reconstruction方向

5.[USC南加州大学 CV Lab](http://www-bcf.usc.edu/~nevatia/index.html) 3D Geometry

6.[Oxford VGG](http://www.robots.ox.ac.uk/~vgg/) 最终的梦想

7.[港中大王晓刚](http://www.ee.cuhk.edu.hk/~xgwang/)有点东西

8.[Kristen Grauman 德克萨斯大学奥斯汀分校](http://www.cs.utexas.edu/~grauman/)

more info[1](https://blog.csdn.net/tercel_zhang/article/details/62883805)[2](https://www.jianshu.com/p/9e08b6fa7a13)


## *2019.4.3*

### *<u>1. facebook和华盛顿大学联合的AR新[研究](https://grail.cs.washington.edu/projects/wakeup/)</u>*

仅仅利用单张图像就可以进行3D姿态重建，使用了Mask-RCNN,骨架结构，还可以让用户修改人物姿态，效果不错。

看这篇博客的[技术解释](http://nooverfit.com/wp/ar%E7%89%88%E7%A5%9E%E7%AC%94%E9%A9%AC%E8%89%AF%EF%BC%9A%E4%BB%8E%E5%8D%95%E5%BC%A02d%E5%9B%BE%E7%89%87%E5%BB%BA%E7%AB%8B3d%E4%BA%BA%E7%89%A9%E8%BF%90%E5%8A%A8%E6%A8%A1%E5%9E%8B/#more-5683)

![structure](https://github.com/Richardyu114/minds-thoghts-and-resources-about-research-/blob/master/images/facebook_AR_3D_SPHOTo.PNG)

### *<u>2. 计算机视觉的3R </u>*

伯克利的大牛Malik最近讲的，计算机视觉包括三个领域（R3），Recognition，Reconstruction & Reorganization。识别和检测只能是第一个R，SLAM算第二个R。图像处理一般算计算机视觉的底层（low level）处理，而增强现实（AR），IBR，计算摄影（computational photography）是计算机视觉和其他领域如图形学，VR，成像学的交集。


### *<u>3. 自动驾驶，计算机视觉科学家[黄浴](https://www.zhihu.com/people/yuhuang2019/answers)在知乎专栏写的一系列博客，综述性很强，值得反复看看</u>*

- [单目视觉深度估计测距的前生今世](https://zhuanlan.zhihu.com/p/56263560)

- [自动/自主泊车技术简介](https://zhuanlan.zhihu.com/p/56236181)

- [SLAM的动态地图和语义问题 （上）](https://zhuanlan.zhihu.com/p/58213757)

- [SLAM的动态地图和语义问题 （下）](https://zhuanlan.zhihu.com/p/58213848)


### *<u> 4.[Computer Vision: Algorithms and Applications](http://szeliski.org/Book/)</u>*

CMU的计算摄影学，其中提到了这个教材，这个教材非常经典，以后有时间挑出来看看吧

[Computational Photography, Fall 2017](http://graphics.cs.cmu.edu/courses/15-463/2017_fall/)

## *2019.4.16*

### *<u>1. 视觉SLAM的面试准备过程以及经典的视觉SLAM算法梳理 </u>*

- [SLAM求职经验帖](https://zhuanlan.zhihu.com/p/28565563)

- [Visual SLAM算法笔记](https://blog.csdn.net/mulinb/article/details/53421864)

- [Annotated Computer Vision Bibliography: Table of Contents](http://www.visionbib.com/bibliography/contents.html)

- [CSE/EE486 Computer Vision I](http://www.cse.psu.edu/~rtc12/CSE486/)+[Multiple View Geometry](https://vision.in.tum.de/teaching/ss2019/mvg2019)

- [TUM computer vision group teaching](https://vision.in.tum.de/teaching/ss2019?redirect=1)


## *2019.4.26*

### *<u>2. 好资源积累  </u>*

1.[blog:The Future of Real-Time SLAM and Deep Learning vs SLAM](http://www.computervisionblog.com/2016/01/why-slam-matters-future-of-real-time.html)

2.[一款矢量图制作工具](https://inkscape.org/)



### *<u>2. 最近进行的学习-----紧急！！！     </u>*

1.面试

python：

https://github.com/taizilongxu/interview_python

机器学习深度学习：

<https://github.com/scutan90/DeepLearning-500-questions>

<https://github.com/imhuay/Algorithm_Interview_Notes-Chinese>

<https://github.com/zeusees/HyperDL-Tutorial>



2.学习：

https://github.com/apachecn/AiLearning

https://github.com/fengdu78/machine_learning_beginner

https://github.com/datawhalechina/pumpkin-book

https://github.com/trekhleb/homemade-machine-learning

https://github.com/Avik-Jain/100-Days-Of-ML-Code

[Introduction to Neural Networks and Machine Learning](http://www.cs.toronto.edu/~tijmen/csc321/information.shtml)

谷歌的机器学习教程和练习


## *2019.4.27*

### *<u>1. 前阿里工作人员撰写的AI[算法笔记](http://www.huaxiaozhuan.com/)  </u>*

内容包含：

- 数学基础

- 统计学习

- 深度学习

- 工具

等等

### *<u>2. 资源 </u>*

1.[Visualizing and Animating Optimization Algorithms with Matplotlib](http://louistiao.me/notes/visualizing-and-animating-optimization-algorithms-with-matplotlib/)

2.[Save Matplotlib Animations as GIFs](http://louistiao.me/posts/notebooks/save-matplotlib-animations-as-gifs/)

3.UCB 2门课程
 - [Stat212b: Topics Course on Deep Learning](http://joanbruna.github.io/stat212b/)
 - [CS294: Deep Reinforcement Learning](http://rail.eecs.berkeley.edu/deeprlcourse/)
