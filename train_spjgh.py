from PIL import ImageFile

# 配置Pillow库来忽略这种错误
ImageFile.LOAD_TRUNCATED_IMAGES = True

from ultralytics import YOLO
import yaml
# Load a model
# model = YOLO("/home/chase/shy/dataset/detect_mohu/yolov8-pose.yaml")  # build a new model from YAML

model = YOLO("/home/chase/shy/dataset/detect_mohu/yolov12/yolo11m-pose.yaml")  # build a new model from YAML
# model = YOLO('/home/chase/shy/dataset/detect_mohu/run/train2/weights/best.pt')  # 加载部分训练的模型
# model = YOLO("yolov8m.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolov8m.yaml").load("yolov8m.pt")  # build from YAML and transfer weights
with open('/home/chase/shy/dataset/detect_mohu/yolov12/default.yaml') as f:
# with open('/home/chase/shy/dataset/detect_mohu/setting.yaml') as f:

    overrides = yaml.safe_load(f.read())
model.train(**overrides)
