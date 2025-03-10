import cv2
import sys
from ultralytics import YOLO
import time
import numpy as np
import json
import base64
import logging
import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"
logging.getLogger("ultralytics").setLevel(logging.CRITICAL) #关闭日志打印
model = YOLO("E:/assets/best.pt",task="detect")
out_results = {
      "status":0,
      "data":[],
      "message":"",
      "timestamp":0,
      "device":device
      }

def model_predict(file_path):
      # 读取图像
      img = cv2.imdecode(np.frombuffer(open(file_path, "rb").read(), np.uint8), cv2.IMREAD_COLOR)
      result = model.predict(source=img,
                              imgsz=1536,
                              conf=0.5,
                              half=True,
                              device=device,
                              save=False,
                              show=False,
                              verbose=False,
                            )
      results = []  # 用于保存结果
      for box in result[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 获取边界框坐标
            width = x2 - x1
            height = y2 - y1
            conf = f"{box.conf[0]:.2f}"   # 获取置信度
            cls_id = int(box.cls[0])  # 获取类别ID
            text = model.names[cls_id]
            results.append({"box": [x1, y1, width, height],"conf": conf,"cls_id": cls_id,"text": text})

      return results
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("ERROR: Invalid arguments")
    else:
         try:
            out_results["timestamp"] = time.time()
            file_paths = base64.b64decode(sys.argv[1]).decode('utf-8')
            json_parse = json.loads(file_paths)
            for i,e in enumerate(json_parse):
                  data = model_predict(e["targetFile"])
                  e["bboxes"] = data
                  out_results["data"].append(e)
            out_results["timestamp"] = int((time.time() - out_results["timestamp"]) * 1000)
            out_results["status"] = 1
            json_data = json.dumps(out_results, ensure_ascii=False, indent=4)
            print(json_data)
         except Exception as e:
            out_results["status"] = 0
            out_results["message"] = f"ERROR in model_predict: {e},{file_paths}"
            print(json.dumps(out_results, ensure_ascii=False, indent=4))

      # out_results["timestamp"] = time.time()
      # file_paths = [{"id":0,"sourceFile":"E:/assets/1.jpg","sliceFile":"E:/assets/1.jpg","targetFile":"E:/assets/1.jpg","bboxes":[]}]
      # for i,e in enumerate(file_paths):
      #             data = model_predict(e["targetFile"])
      #             e["bboxes"] = data
      #             out_results["data"].append(e)
      # out_results["timestamp"] = int((time.time() - out_results["timestamp"]) * 1000)
      # out_results["status"] = 1
      # print(out_results)