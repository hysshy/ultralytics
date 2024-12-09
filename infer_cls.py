import cv2
import os
import time
import torch
import shutil
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def getAllFile(picPath, key):
    picList = []
    for root, dirs, files in os.walk(picPath):
        for file in files:
            if key in file:
                picList.append(os.path.join(root,file))
    return picList

savepath = '/home/chase/shy/dataset/colorClassify/draw'  # 姿态保存路径
imgpath = '/home/chase/shy/dataset/colorClassify/val'  # 待预测图片路径
from ultralytics import YOLO
# Load a model
model = YOLO('/home/chase/shy/dataset/colorClassify/train2/weights/best.onnx', task='classify') # 加载部分训练的模型

piclist = getAllFile(imgpath, 'jpg')  # 获取图片列表
for imgfile in piclist:
    img = cv2.imread(imgfile)
    start = time.time()
    result = model(img, device=device, imgsz=160) # 预测图片
    label = result[0].probs.top1
    labelName = result[0].names[label]
    print(time.time() - start)
    os.makedirs(os.path.join(savepath, labelName), exist_ok=True)
    shutil.copy(imgfile, os.path.join(savepath, labelName))  # 保存图片到对应类别文件夹
