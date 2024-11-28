from PIL import ImageFile
import cv2
import os
import time
import torch
from torch.xpu import device

savepath = '/home/chase/shy/dataset/test'  # 姿态保存路径
imgpath = '/home/chase/shy/dataset/detect_zitai/val/images/detect'  # 待预测图片路径
from ultralytics import YOLO
import yaml
# Load a model
# model = YOLO("/home/chase/shy/dataset/detect_mohu/yolov8-pose.yaml")
# model = YOLO('/home/chase/shy/dataset/detect_mohu/run2/train14/weights/best.pt') # 加载部分训练的模型
model = YOLO('/home/chase/shy/dataset/detect_mohu/run2/train18/weights/best.onnx', task='pose') # 加载部分训练的模型
# device = torch.device('cuda:0')
# model.predictor.device = device
for imgName in os.listdir(imgpath):
    img = cv2.imread(imgpath + '/' + imgName)
    start = time.time()
    result = model(img, device='cuda:0')  # 预测图片
    print(time.time() - start)
    # result = model('/home/chase/shy/dataset/detect_zitai/val/images/detect/jiaoyan1_44599.825625_1370649456974999552.jpg')
    bboxes = result[0].boxes.xyxy.cpu().numpy()  # 预测框
    cls = result[0].boxes.cls.cpu().numpy()  # 预测类别
    names = result[0].names  # 类别名称
    keypoints = result[0].keypoints.xy.cpu().numpy()  # 预测关键点
    img = result[0].orig_img  # 原始图片
    imgName = result[0].path.split('/')[-1]  # 图片名称
    zitai = result[0].zitai.cpu().numpy()  # 预测的姿态
    mohus = result[0].mohu.cpu().numpy()  # 预测的姿态
    for i in range(len(bboxes)):
        label = names[cls[i]] # 标签
        bbox = bboxes[i].astype(int) # 预测框
        if label in ['face']:
            mohu = mohus[i][0] # 模糊度
            mohu_savepath = savepath + '/mohu' + str(round(mohu,2)) # 保存路径
            if not os.path.exists(mohu_savepath):
                os.makedirs(mohu_savepath)
            cv2.imwrite(mohu_savepath +'/'+ imgName.replace('.jpg', '_' + str(i) + '.jpg'), img[bbox[1]:bbox[3], bbox[0]:bbox[2]]) # 保存图片

        if label in ['face', 'facewithmask']:
            zitaiLabel = names[zitai[i][0]+9] # 姿态标签
            if not os.path.exists(savepath +'/'+ zitaiLabel):
                os.makedirs(savepath +'/'+ zitaiLabel)
                cv2.imwrite(savepath +'/'+ zitaiLabel + '/' + imgName.replace('.jpg', '_' + str(i) + '.jpg'), img[bbox[1]:bbox[3], bbox[0]:bbox[2]]) # 保存图片
            kp = keypoints[i].astype(int) # 预测关键点
            for j in range(kp.shape[0]):
                cv2.circle(img, (kp[j][0], kp[j][1]), 2, (0, 0, 255), -1) # 画关键点
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2) # 画框
        cv2.putText(img, label, (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) # 写标签
    cv2.imwrite(savepath +'/'+ imgName, img) # 保存图片
    #result[0].show()  # 显示预测结果
    # while True:
    #     start = time.time()
    #     result = model('/home/chase/shy/dataset/detect_kp/val/images/detect/1592259996533.jpg')  # 预测图片
    #     print(time.time() - start)
    # result[0].show()  # 显示预测结果