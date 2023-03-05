from ultralytics import YOLO
from ultralytics.yolo import v8, data
model = YOLO('yolov8n.pt')
results = model.train(data='coco128.yaml', epochs=100, imgsz=600)
print(results)