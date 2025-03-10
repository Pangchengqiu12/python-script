# Ultralytics YOLO 🚀, AGPL-3.0 license
import sys
import cv2
import numpy as np
import onnxruntime as ort
import base64
import json
import time
out_results = {
    "status": 0,
    "data": [],
    "message": "",
    "timestamp": 0,
    "device":""
}

class YOLO11:
    """YOLO11 目标检测模型类，用于处理推理和可视化。"""
    def __init__(self, onnx_model, input_image, confidence_thres, iou_thres):
        """
        初始化 YOLO11 类的实例。
        参数：
            onnx_model: ONNX 模型的路径。
            input_image: 输入图像的路径。
            confidence_thres: 用于过滤检测结果的置信度阈值。
            iou_thres: 非极大值抑制（NMS）的 IoU（交并比）阈值。
        """
        self.onnx_model = onnx_model
        self.input_image = input_image
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres

        # 加载类别名称
        self.classes = {}

        # 为每个类别生成一个颜色调色板
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def preprocess(self,img):
        """
        对输入图像进行预处理，以便进行推理。
        返回：
            image_data: 经过预处理的图像数据，准备进行推理。
        """
        # 使用 OpenCV 读取输入图像
        self.img = cv2.imread(img)
        # 获取输入图像的高度和宽度
        self.img_height, self.img_width = self.img.shape[:2]

        # 将图像颜色空间从 BGR 转换为 RGB
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        # 保持宽高比，进行 letterbox 填充, 使用模型要求的输入尺寸
        img, self.ratio, (self.dw, self.dh) = self.letterbox(img, new_shape=(self.input_width, self.input_height))

        # 通过除以 255.0 来归一化图像数据
        image_data = np.array(img) / 255.0

        # 将图像的通道维度移到第一维
        image_data = np.transpose(image_data, (2, 0, 1))  # 通道优先

        # 扩展图像数据的维度，以匹配模型输入的形状
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        # 返回预处理后的图像数据
        return image_data


    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
        """
        将图像进行 letterbox 填充，保持纵横比不变，并缩放到指定尺寸。
        """
        shape = img.shape[:2]  # 当前图像的宽高

        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # 计算缩放比例
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])  # 选择宽高中最小的缩放比
        if not scaleup:  # 仅缩小，不放大
            r = min(r, 1.0)

        # 缩放后的未填充尺寸
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))

        # 计算需要的填充
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # 计算填充的尺寸
        dw /= 2  # padding 均分
        dh /= 2

        # 缩放图像
        if shape[::-1] != new_unpad:  # 如果当前图像尺寸不等于 new_unpad，则缩放
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        # 为图像添加边框以达到目标尺寸
        top, bottom = int(round(dh)), int(round(dh))
        left, right = int(round(dw)), int(round(dw))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        return img, (r, r), (dw, dh)



    def postprocess(self, output):
        """
        对模型输出进行后处理，以提取边界框、分数和类别 ID。
        参数：
            input_image (numpy.ndarray): 输入图像。
            output (numpy.ndarray): 模型的输出。
        返回：
            numpy.ndarray: 包含检测结果的输入图像。
        """
        # 转置并压缩输出，以匹配预期形状
        outputs = np.transpose(np.squeeze(output[0]))
        rows = outputs.shape[0]
        boxes, scores, class_ids = [], [], []

        # 计算缩放比例和填充
        ratio = self.img_width / self.input_width, self.img_height / self.input_height

        for i in range(rows):
            classes_scores = outputs[i][4:]
            max_score = np.amax(classes_scores)
            if max_score >= self.confidence_thres:
                class_id = np.argmax(classes_scores)
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # 将框调整到原始图像尺寸，考虑缩放和填充
                x -= self.dw  # 移除填充
                y -= self.dh
                x /= self.ratio[0]  # 缩放回原图
                y /= self.ratio[1]
                w /= self.ratio[0]
                h /= self.ratio[1]
                left = int(x - w / 2)
                top = int(y - h / 2)
                width = int(w)
                height = int(h)
                x1 = left
                y1 = top
                x2 = left + width
                y2 = top + height
                boxes.append([x1, y1, x2, y2])
                scores.append(max_score)
                class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)
        result = []
        for i in indices:
            box = boxes[i]
            score = round(float(scores[i]), 2)
            class_id = int(class_ids[i])
            class_name = self.classes[class_id]
            result.append({"box":box,"score":score,"classId":class_id,"className":class_name})
        return result



    def main(self):
        providers=["DmlExecutionProvider", "CPUExecutionProvider"] if ort.get_device() == "CPU-DML" else ["CPUExecutionProvider"]
        out_results["device"] = providers
        # 使用 ONNX 模型创建推理会话，自动选择CPU或GPU
        session = ort.InferenceSession(
            self.onnx_model,
            providers=providers
        )
        meta = session.get_modelmeta().custom_metadata_map
        label_names = eval(meta['names']) # 获取标签名
        self.classes = label_names

        # 获取模型的输入形状
        model_inputs = session.get_inputs()
        input_shape = model_inputs[0].shape
        self.input_width = input_shape[2] # 模型输入宽度
        self.input_height = input_shape[3] # 模型输入高度

        results = []
        for i, e in enumerate(self.input_image):
            # 预处理图像数据，确保使用模型要求的尺寸 (640x640)
            img_data = self.preprocess(e["targetFile"])

            # 使用预处理后的图像数据运行推理
            outputs = session.run(None, {model_inputs[0].name: img_data})
            data = self.postprocess(outputs[0])
            e["status"] = 1
            e["bboxes"] = data
            results.append(e)
        return results





if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("ERROR: Invalid arguments")
    else :
        out_results["timestamp"] = time.time()
        imgs =  json.loads(base64.b64decode(sys.argv[1]).decode('utf-8'))
        model = "E:/assets/best.onnx"
        conf_thres = 0.5
        iou_thres = 0.5
        # 使用指定的参数创建 YOLO11 类的实例
        detection = YOLO11(model, imgs, conf_thres, iou_thres)

        # 执行目标检测并获取输出图像
        data = detection.main()
        out_results["data"] = data
        out_results["timestamp"] = int((time.time() - out_results["timestamp"]) * 1000)
        json_data = json.dumps(out_results, ensure_ascii=False, indent=4)
        print(json_data)