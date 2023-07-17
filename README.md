# Paddle-Traffic-statistics
通过人体跟踪实现人流量统计
这是一种关于训练人体目标检测模型的深度学习方法，该方法部署于百度飞桨平台（paddlepaddle）经笔者测试，识别度能达到94%以上。



环境配置：

docker部署 

python：3.7.13

PaddlePaddle 2.4.2


部署环境：
先拉paddle镜像
```
docker pull 2.4.2-gpu-cuda11.2-cudnn8.2-trt8.0
```
再拉PaddleDetection源码
```
git clone https://gitee.com/paddlepaddle/PaddleDetection.git
```
运行容器
```
nvidia-docker run -it --privileged=true --name paddle_test --gpus all --shm-size 8g -d  -p 8040:8040 -v /home/yd/PaddleDetection:/PaddleDetection paddlepaddle/paddle:2.4.2-gpu-cuda11.2-cudnn8.2-trt8.0 /bin/bash
```

数据集使用是的是MOT17,相对于MOT16多了部分数据,根据自身需求选择数据集
```
wget https://motchallenge.net/data/MOT17.zip
```
使用 configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml配置文件

根据自己的文件路径修改配置文件信息，目前是单目标跟踪，自己可对照更改
```
metric: MOT
num_classes: 1

# for MOT training
TrainDataset:
  !MOTDataSet
    dataset_dir: dataset/mot
    image_lists: ['mot17.train', 'caltech.all', 'cuhksysu.train', 'prw.train', 'citypersons.train', 'eth.train']
    #image_lists: ['mot17.train']
    data_fields: ['image', 'gt_bbox', 'gt_class', 'gt_ide']

# for MOT evaluation
# If you want to change the MOT evaluation dataset, please modify 'data_root'
EvalMOTDataset:
  !MOTImageFolder
    dataset_dir: dataset/mot
    data_root: MOT17/images/train
    keep_ori_im: False # set True if save visualization images or video, or used in DeepSORT

# for MOT video inference
TestMOTDataset:
  !MOTImageFolder
    dataset_dir: dataset/mot
    keep_ori_im: True # set True if save visualization images or video

```


