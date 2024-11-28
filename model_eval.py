from PIL import ImageFile
import cv2
import os
import time
from ultralytics import YOLO

model = YOLO('/home/chase/shy/dataset/detect_mohu/run/train2/weights/best.pt') # 加载部分训练的模型
model.val()
