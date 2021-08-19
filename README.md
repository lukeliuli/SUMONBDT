# SUMONBDT
将NBDT和SUMO结合起来，用于车道选择  
##对NBDT的基本理解  
* NBDT,前端CNN后面DT.  
* DT诱导层的叶子节点来源于预先已经训练好神经网络（以下简称网A）最后一层FC层的权重。注意：这个权重是在NBDT训练期间是不变的  
* 其余诱导层的原理和概念都比较好理解,百度其他文章，例如https://developer.aliyun.com/article/763702  
* NBDT的训练期间，是更新前端CNN的网络权重(和经典神经网络学习没有任何区别)。
* 注意：在代码中，前端CNN和网络A一致，但是在训练过程中，前端CNN的最后一层FC层的权重变化了，不会影响DT。在整个训练过程中诱导层的叶子节点不变化，DT任何都不会改变,包括叶子节点不变化
* 简单解释一下，预训练网络A用于提取特征权重。这个特征权重给了DT以后就不变化了。前端网络CNN用于提取实际输入图像的特征，而训练前端网络CNN(DT不训练)迫使模型对新样本的特征提取能够遵循树结构从根节点一路推断至叶节点，也就是finetune。
* 看原始文献比较好理解作者想法和思路，例如为什么采用WORDNET,原因是作者需要一个建造DT的合理省事的方法（不是最优的方法甚至是次优） ，而且要说服审稿人
* 这里对NBDT的不成熟看法，对前端CNN的训练就是作者想不出其他办法了(DT已经固定了)，暴力性的重新训练前端CNN,让加入DT的NBDT（CNN+DT）准确性提高。反正DT已经负责解释性了。然后他成功了。  
## 代码首先是最NBDT的简化  
* mainNBDT1.py是基于pytorch的CIFA训练。包括各种加速技巧和提高准确率的技巧，；例如CUBA,变学习率，增加数据增强
* 注意1： 用torvision的resnet模型，最多到88%的正确率，但是用https://github.com/WeihaoZhuang/cifar10-100-fast-training的模型能够到95%
* 注意2：参考https://github.com/WeihaoZhuang/cifar10-100-fast-training 训练速度能变快 
* testNetworkxWordnet用于学习networkx和wordnet的程序  