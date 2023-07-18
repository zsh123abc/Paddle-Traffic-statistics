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

训练模型
```python
python -m paddle.distributed.launch --log_dir=./fairmot_dla34_30e_1088x608/ --gpus 0,1,2  tools/train.py -c configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml  --use_vdl=true --vdl_log_dir=vdl_dir/fai --eval>out.log 2>&1&
```
可视化查看训练过程，前提是按照了visualdl，用pip安装
```shell
visualdl --logdir=vdl_dir/fairmot_dla34_30e_1088x608/ --host 0.0.0.0 &
```

模型导出
```
!CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml -o weights=output/fairmot_dla34_30e_1088x608/model_final.pdparams
```

优化模型，转换格式
```python
paddle_lite_opt --valid_targets=arm \
--model_file=inference_model/fairmot_dla34_30e_1088x608/model.pdmodel \
--param_file=inference_model//fairmot_dla34_30e_1088x608.pdiparams \
--optimize_out=inference_model/fairmot_dla34_30e_1088x608/ping-pang
```

模型预测视频,需ffmpeg，ffmpeg自行下载
```
CUDA_VISIBLE_DEVICES=0 python tools/infer_mot.py -c configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml -o weights=output/fairmot_dla34_30e_1088x608/5.pdparams   --video_file=dataset/video/sc2.mp4  --frame_rate=20 --save_videos
```
