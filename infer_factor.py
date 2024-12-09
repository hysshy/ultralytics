import cv2
import os
import time
savepath = '/home/chase/shy/dataset/test2'  # 姿态保存路径
imgpath = '/home/chase/shy/dataset/bodyfactor_detect/val/images'  # 待预测图片路径
from ultralytics import YOLO
# Load a model
model = YOLO('/home/chase/shy/dataset/bodyfactor_detect/run/train21/weights/last.pt', task='detect') # 加载部分训练的模型

for imgName in os.listdir(imgpath):
    img = cv2.imread(imgpath + '/' + imgName)
    start = time.time()
    result = model(img, device='cuda:0', conf=0.25, imgsz=160)  # 预测图片
    print(time.time() - start)
    bboxes = result[0].boxes.xyxy.cpu().numpy()  # 预测框
    cls = result[0].boxes.cls.cpu().numpy()  # 预测类别
    names = result[0].names  # 类别名称
    img = result[0].orig_img  # 原始图片
    # imgName = result[0].path.split('/')[-1]  # 图片名称
    # print(imgName)
    for i in range(len(bboxes)):
        label = names[cls[i]] # 标签
        bbox = bboxes[i].astype(int) # 预测框
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)  # 画框
        cv2.putText(img, label, (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # 写标签
    cv2.imwrite(savepath + '/' + imgName, img)  # 保存图片