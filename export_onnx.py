from PIL import ImageFile
import cv2
import os
import time
import torch
from torch.xpu import device

from ultralytics import YOLO
model = YOLO('/home/chase/shy/dataset/colorClassify/train2/weights/best.pt', task='classify') # 加载部分训练的模型
model.export(format='onnx')
