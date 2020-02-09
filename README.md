# Towards-Realtime-MOT
本项目主要为本人对[Towards-Realtime-MOT](https://arxiv.org/pdf/1909.12605v1.pdf)论文中代码的一个理解，原项目主页为[Github](https://github.com/Zhongdao/Towards-Realtime-MOT).

实际上该项目主要是YOLO和Deep sort的结合体，在训练阶段基本上是YOLO，只是加了一个对id的embedding
在track阶段主要是Deep sort，所以为了更好的理解可以阅读有关Deep sort的说明。
关于Deep sort的核心是匈牙利树和卡尔曼滤波：
每个要跟踪的对象都是一个卡尔曼滤波，卡尔曼滤波的核心是概率论，是均值和误差。
卡尔曼滤波不像是高斯滤波之类的，主要是基于测量值和估计值的一个迭代， 
卡尔曼滤波会有一个预测的anchor,而检测器会有一个测量值，即yolo的detection，然后用detection更新卡尔曼滤波器参数，以使得更好的预测
然后匈牙利树就是一个匹配问题，匹配上一帧和当前帧，可以通过将帧之间的距离(embedding和iou)作为代价，求解代价最小情况下的匹配，

对于unconfirmed tracks，如果没有匹配到detection，则应该移除
对于confirmed tracks，如果没有匹配到detection，则应该视为丢失

本人在该项目的理解上做了一点注释，并做了一点点修改，只是为了方便阅读。

本人已标注除了track的大部分代码。track阶段对于匈牙利树还有卡尔曼滤波阶段的知识并不是很了解，仅仅是在大致清楚输入和输出的情况下做了部分标注。


参考文献：  

1： [知乎大神](https://zhuanlan.zhihu.com/p/90835266)，这个博客中有匈牙利树的求解方法很形象，还有卡尔曼滤波的几个公式
2： [csdn大神](https://blog.csdn.net/sgfmby1994/article/details/98517210)，这个博客对track阶段的流程做了很详细的解释
