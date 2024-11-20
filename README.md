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
* mainNBDT1.ipynb是在colab.google运行的NBDT+CIFAR程序，与mainNBDT1.py大概率一致。如果不一致的话，colab.google有最新代码

# 最新更新
* my1Lane1TlsVeh6-server.sumocfg 和 my1Lane1TlsVeh6.rou.xml 用于调节模拟中的车辆和车辆类型参数，并用GUI进行验证
* mainTest2_hmcnf_keras_dt为mainTestCSVMLP3(hmcnf_keras)的简化和优化代码，代码清晰化和优化
* 运行环境为conda activate tensor23py36cpu
* branch_2slot_5sep 为主程序修改程序。 (1) 输入数据为前后2时刻的2slot，(2) 模型的时间分割为5公里每小时，
* branch_2slot_5sep 主程序mainTest2_hmcnf_keras_dt_2slot_5sep 和 extractFranceSamples1_2slot_5sep.ipynb
* 发现HMCN-F 训练效果很差，所以改为分离样式多层训练主程序mainTest2_hmcnf_keras_dt_2slot_5sep
* mainSimpleStep3为简化的步骤3程序，注意只训练最底层1层模型,用于分析模型和误差
* mainSimpleStep4为简化的步骤4程序，注意只能接mainSimpleStep3的输出结果，用于分析模型和误差
---

* 注意！ 注意！ 注意！ 注意！
* branch_2slot_5sep 为主程序所在地
* branch_2slot_5sep 主程序为mainTest2_hmcnf_keras_dt_2slot_5sep 和 extractFranceSamples1_2slot_5sep.ipynb
*   extractFranceSamples1_2slot_5sep.ipynb 将法国的数据库转换为模型所需要的数据集合 ：（0）生成为输入数据为前1时刻的样本 （1）转换样本为输入数据为前2时刻的样本
*   mainTest2_hmcnf_keras_dt_2slot_5sep_siat 输入数据为前2时刻的样本并建立层次结构，并进行训练和识别
* step0获得:原始数据库数据（有一定挑选）'''
* step1获得:根据step0的识别模型和识别结果的从step0中样本挑选低概率样本xlowpra'''
* step2获得：step1中低概率样本中SUMO成果仿真获取的样本（有一定挑选，去掉无法仿真样本大概1%），其实作用不大，只是为了发论文
* step3获得：将step2的样本与对应的step1的低概率样本的进行合成，并训练加强模型stage2Mode    ， 其实作用不大，只是为了发论文                                          
* step3找到step2样本（就是df_step2）在step1的样本（xlowpra1）相对应的样本xlowpra2。然后df_step2和xlowpra2合成为x,用于训练加强模型stage2Mode。其实作用不大，只是为了发论



# SUMO+PYTORCH+NBDT的配置命令
sudo apt-get update  
sudo apt-get upgrade  
https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh  
conda update anaconda
conda env list  
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/  

conda create -n pytorch  python=3.6  

conda activate pytorch  
conda install pytorch torchvision cudatoolkit=11.0  
conda install jupyter notebook  
conda install nb_conda  
conda install -c conda-forge jupyter_contrib_nbextensions  
conda install -c conda-forge jupyter_nbextensions_configurator  
jupyter notebook --generate-config  
jupyter notebook password  
c.NotebookApp.allow_remote_access = True  
c.NotebookApp.ip = '*' # 所有绑定服务器的IP都能访问，若想只在特定ip访问，输入ip地址即可  
c.NotebookApp.open_browser = False # 我们并不想在服务器上直接打开Jupyter Notebook，所以设置成False  
c.NotebookApp.port = 7777 # 将端口设置为自己喜欢的吧，默认是8888  
c.NotebookApp.notebook_dir = '/home/liuli'  


 conda install pandas  
 pip3 install sklearn  
https://sumo.dlr.de/docs/Installing/Linux_Build.html   
pip install --upgrade pip   
conda install git cmake python3 g++ libxerces-c-dev libfox-1.6-dev libgdal-dev libproj-dev libgl2ps-dev python3-dev swig default-jdk maven   libeigen3-dev  
 git clone --recursive https://github.com/eclipse/sumo  
mv sumo-main sumo  
export SUMO_HOME="$PWD/sumo"  
 mkdir sumo/build/cmake-build && cd sumo/build/cmake-build  
 cmake ../..  
 make -j4  
sudo apt-get install libavformat-dev libswscale-dev libopenscenegraph-dev python3-dev swig libgtest-dev libeigen3-dev python3-pip python3-setuptools default-jdk  

ssh-keygen -o  
cat ~/.ssh/id_rsa.pub  
git clone git@github.com:lukeliuli/SUMONBDT.git  

# 用于win10环境下，tensorflow-gpu的建立
首先win10下面安装cuba,cudnn,并配置环境变量  
conda install cudatoolkit=10.1 cudnn=7.6.5         
pip install tensorflow-gpu==2.3.0 -i https://pypi.tuna.tsinghua.edu.cn/simple  
pip install keras==2.3.1 -i https://pypi.tuna.tsinghua.edu.cn/simple  
https://blog.csdn.net/edward_zcl/article/details/124543504    
https://blog.csdn.net/QAQIknow/article/details/118858870  

有用的的命令  
conda create -n gpuKeras2 python=3.8

conda install jupyter notebook  

jupyter notebook --generate-config  
jupyter notebook password  
c.NotebookApp.allow_remote_access = True  
c.NotebookApp.ip = '*' # 所有绑定服务器的IP都能访问，若想只在特定ip访问，输入ip地址即可  
c.NotebookApp.open_browser = False # 我们并不想在服务器上直接打开Jupyter Notebook，所以设置成False  
c.NotebookApp.port = 7777 # 将端口设置为自己喜欢的吧，默认是8888  
c.NotebookApp.notebook_dir = '/home/liuli'  


conda install nb_conda
conda install pandas  
pip3 install sklearn 
 
conda install -c conda-forge imbalanced-learn
conda install matplotlib 



# docker 基本命令
1. 看docker运行实例: sudo docker ps
2. 以命令行登录docker实例,sudo docker exec -it t1  /bin/bash
3. 或者第一次运行

sudo docker run -itd --name t1 -v /home/liuli:/home/liuli0 -p 80:8012  -p 22:8022 sumoquaninvestjupyter:1.7   /bin/bash 
sudo docker run -itd sumoquaninvestjupyter:1.7   /bin/bash

#### docker 映射端口
+ sudo docker run -itd -p 6666:6666  laughing_kalam  /bin/bash

####docker 停止
+ docker stop laughing_kalam

删除所有未运行的容器
docker rm $(docker ps -a -q)


#### docker commit :从容器创建一个新的镜像
+ docker commit <容器id> <镜像名>:<tag>
+ 例如 sudo docker commit t1 sumoquaninvestjupyter:1.7

#### 将指定镜像保存成 tar 归档文件
+ docker save -o ubuntuSumoQuaninvestJupyterV1.tar sumoquaninvestjupyter:1.0

#### 从 tar 导入镜像
+ docker load -i ubuntuSumoQuaninvestJupyterV1.tar


#### 查看镜像
+ sudo docker images 

#### 删除单个镜像
+ sudo docker rmi -f  <镜像id>
    
#### pydot画神经网络图
pip install pydot
pip install pydotplus 
conda install pydot  