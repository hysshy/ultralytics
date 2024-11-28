from PIL import ImageFile
import cv2
import os
import time
import torch
from torch.xpu import device

from ultralytics import YOLO
model = YOLO('/home/chase/shy/dataset/detect_mohu/run2/train18/weights/best.pt', task='pose') # 加载部分训练的模型
model.export(format='onnx')
