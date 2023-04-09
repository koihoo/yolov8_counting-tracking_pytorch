# yolov8 目标检测+计数+追踪
---

* 实现目标检测中的 图片/视频 计数
* 显示检测的类别
* 追踪id

### 效果展示

![节点](https://cdn.nlark.com/yuque/0/2023/png/35529404/1680835735502-8fabe78c-9e87-4ec8-b615-c2927eb4eb56.png?x-oss-process=image%2Fresize%2Cw_632%2Climit_0) 

### 一、运行环境 
* ubuntu 18.04
* python 3.8
* pytorch 1.11

### 二、环境安装
2.1、下载代码 进入相应目录
```
git clone https://github.com/koihoo/yolov8_counting-tracking_pytorch.git
cd yolov8_counting-tracking_pytorch
```
2.2、conda创建虚拟环境 并激活环境
```
conda create -n yolov8 python=3.8
conda activate yolov8
```
2.3、安装pytorch
> 根据操作系统、安装工具以及CUDA版本，在 https://pytorch.org 找到对应的安装命令。我的环境是 ubuntu 18.04.5、pip、CUDA 11.3
```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```
2.4、安装必要的软件包
```
pip install ultralytics
```
### 三、数据准备——训练自己的数据集
3.1 原始数据格式(插入yolov8/data文件夹中)
```
dataset #(数据集名字：例如goose) 
├── images      
       ├── train          
              ├── xx.jpg     
       ├── val         
              ├── xx.jpg 
├── labels      
       ├── train          
              ├── xx.txt     
       ├── val         
              ├── xx.txt 
```

在yolov8/data文件夹下新建goose.yaml

```
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]


path: /home/koihoo/yolov8/data/goose # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')
test:  # test images (optional)

# Classes
nc: 1  # number of classes
names: ['goose']  # class names
```

>path：数据集的根目录
train：训练集与path的相对路径
val：验证集与path的相对路径
nc：类别数量，因为这个数据集只有一个类别（fire），nc即为1。
names：类别名字

3.2下载预训练模型	
根据需求选择 https://github.com/ultralytics/ultralytics 
模型下载完成后，将xx.pt复制在yolov8_counting-tracking_pytorch文件夹下。

### 四、训练
上面的数据和预训练模型都准备好之后，我们就可以开始训练啦

方法一
* yolov8可以直接用CIL命令行开始训练：
```
yolo task=detect mode=train model=yolov8x.pt data=data/goose.yaml batch=16 epochs=100 igsz=640 workers=8 device=0 
```

方法二
* 采用python脚本的方式
```
from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8x.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="coco128.yaml", epochs=3)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
# success = model.export(format="onnx")  # export the model to ONNX format
```
训练最终结果如图所示

![Alt text](https://cdn.nlark.com/yuque/0/2023/png/35529404/1679989736229-3a4253d4-23f7-4469-82fe-70d619a42e8c.png)

> 训练窗口中添加需要打印的指标（如F1-score）
 
todo


### 五、测试(video tracking + counting)
目前yolov8的追踪模型支持：
  *  ByteTracker
  *  BoT-SORT

采用python脚本的方式执行追踪功能：在/home/koihoo/ultralytics/track_v8.py
```
from ultralytics import YOLO

model = YOLO("/home/koihoo/yolov8/runs/detect/train7/weights/best.pt")
model.track(
    source="/home/koihoo/yolov8/data/goose-video/TundraSwan2.mp4",
    # stream=True,
    tracker="/home/koihoo/ultralytics/ultralytics/tracker/cfg/botsort.yaml",  # or 'bytetrack.yaml'
    save = True,
    conf = 0.6,
    hide_conf = True,
)
```

> 在xx/ultralytics/yolo/v8/predict.py中加入counting的功能代码

```python\
# L 78
####### COUNTING ########
cv2.putText(im0,f"{names[int(c)]}{'s' * (n > 1)}: {n}", (5,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
```

---
# 使用框架
* https://github.com/ultralytics/ultralytics/