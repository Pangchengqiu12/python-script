import os
import shutil
import time
import gc
import cv2
import uuid
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import queue
import threading
from datetime import datetime, timedelta
# import ffmpeg
import imageio.v2 as imageio
import psutil
import subprocess as sp
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pydub import AudioSegment
import librosa
import requests
import torch
from transformers import pipeline
import tempfile
from flask import Flask, request, jsonify
from collections import OrderedDict
import matplotlib.pyplot as plt
from flask.json import dumps
from flask_cors import CORS
from gevent import pywsgi
import base64
from mmdet.apis import inference_detector, init_detector
from mmyolo.utils import switch_to_deploy
from mmdeploy.utils import get_input_shape, load_config
from mmdeploy.apis.utils import build_task_processor
import xlsxwriter
from categories import bird100_categories, bird100_palette

import warnings
warnings.filterwarnings("ignore")


'''
This script is used to deploy the model using gevent WSGI server.
'''

app = Flask(__name__)

CORS(app)


'''
INITIALIZATION
- Load the models for bird sound recognition and bird image detection.
'''

# Heartbeat
client_heartbeats = {}  # 存放客户端心跳时间和状态
DEAD_THRESHOLD = 15  # 心跳容忍时间
STOP_STREAM_TRIGGERED = False  # 控制 stop_stream 是否已经触发

# For Video Bird Detection
font_path = 'statics/MSYH.TTC'
config_file = '/filedata/AiModel/model/mmyolo/yolov8_s_video_direct_pretrained_size-1536_librated-scale_amp/yolov8_s_video_direct_pretrained_size-1536_librated-scale_amp.py'
checkpoint_file = '/filedata/AiModel/model/mmyolo/yolov8_s_video_direct_pretrained_size-1536_librated-scale_amp/best_coco_bbox_mAP_epoch_496.pth'
tensorrt_checkpoint_file = '/filedata/AiModel/model/yolov8_tensorrt_fp16/end2end.engine'
tensorrt_config_file = 'mmyolo_for_deploy/configs/deploy/detection_tensorrt_static-1536x1536.py'
yolov11_checkpoint_file = '/filedata/AiModel/model/ultralytics/yolov11/best.pt'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
font = ImageFont.truetype(font_path, 45)

model = init_detector(config_file, checkpoint_file, device=device) # inference_detector
switch_to_deploy(model)

deploy_cfg, model_cfg = load_config(tensorrt_config_file, config_file)# tensorrt_model
task_processor = build_task_processor(model_cfg, deploy_cfg, device)
tensorrt_model = task_processor.build_backend_model([tensorrt_checkpoint_file])
input_shape = get_input_shape(deploy_cfg)

yolov11_model = YOLO(yolov11_checkpoint_file)
print('[VIDEO] Model loaded')

results_store = {} # Store to keep track of video processing jobs
video_streams = {} # Store to keep track of video streams
video_queue = queue.Queue() # Queue to store video tasks
detection_queue = Queue() # Queue to store detection tasks
sound_queue = Queue() # Queue to store sound tasks

# For Audio Bird Detection
[os.remove(os.path.join('/filedata/AiModel/tmp', file)) for file in os.listdir('/filedata/AiModel/tmp')]
pipe = pipeline(task="audio-classification", model="/filedata/AiModel/model/transformers/ast/checkpoint-10543", device='cuda')
print('[SOUND] Model loaded')

# Callbock
callback_url = "http://localhost:59200/callback/recognizeResult"

# release resources
MAX_MEMORY_USAGE_MB = 1500


'''
HEARTBEAT
'''
# Declare the global variable
last_heartbeat = datetime.now()  # Initialize heartbeat timestamp globally

# 检查心跳超时的线程
def check_heartbeats():
    global STOP_STREAM_TRIGGERED, last_heartbeat  # Use the global last_heartbeat

    while True:
        now = datetime.now()

        # If the time since the last heartbeat exceeds the threshold and stop_stream not triggered yet
        if now - last_heartbeat > timedelta(seconds=DEAD_THRESHOLD) and not STOP_STREAM_TRIGGERED:
            with app.app_context():
                STOP_STREAM_TRIGGERED = True
                stop_stream()  # Call stop_stream when the heartbeat is dead

        # Check heartbeat every 1 second
        time.sleep(1)

@app.route('/heartbeat', methods=['POST'])
def heartbeat():
    global STOP_STREAM_TRIGGERED, last_heartbeat  # Use the global last_heartbeat

    # Update the global heartbeat timestamp
    last_heartbeat = datetime.now()
    STOP_STREAM_TRIGGERED = False  # Reset the trigger flag on new heartbeat

    return {'status': 'alive'}, 200, {'Content-Type': 'application/json'}


'''
PUBLIC API
'''
@app.route('/check_status/<job_id>', methods=['GET'])
def get_status(job_id):
    if job_id not in results_store:
        return jsonify({'error': 'Invalid job ID'}), 404, {'Content-Type': 'application/json'}

    return jsonify(results_store[job_id]), {'Content-Type': 'application/json'}

@app.route('/cancel_job/<job_id>', methods=['POST'])
def cancel_specific_recognition(job_id):
    if job_id not in results_store:
        return jsonify({'error': 'Invalid job ID', 'status': 'failed'}), 404, {'Content-Type': 'application/json'}

    if results_store[job_id]['status'] == 'completed':
        return jsonify({'error': 'Cannot cancel a completed task', 'status': 'failed'}), 400, {'Content-Type': 'application/json'}

    results_store[job_id]['cancelled'] = True
    return jsonify({'job_id': job_id, 'status': 'cancelled'}), 200, {'Content-Type': 'application/json'}


'''
PUBLIC FUNCTIONS
'''
def estimate_remaining_time(start_time, current_time, progress):
    elapsed_time = current_time - start_time
    if progress <= 0:
        return 'Calculating...'
    estimated_total_time = elapsed_time / progress
    remaining_time = estimated_total_time - elapsed_time
    return f'{round(remaining_time, 2)} 秒剩余'

def check_and_clear_results_store():
    # 获取当前内存使用情况
    process = psutil.Process(os.getpid())
    memory_usage_mb = process.memory_info().rss / 1024 / 1024  # 转换为 MB

    if memory_usage_mb > MAX_MEMORY_USAGE_MB:
        print(f"[MEMORY] Memory usage is {memory_usage_mb:.2f}MB, clearing non-processing results.")
        # 清理 'status' 不为 'processing' 的成员
        to_remove = [job_id for job_id, result in results_store.items() if result.get('status') != 'processing']
        for job_id in to_remove:
            del results_store[job_id]
        print(f"[MEMORY] Cleared {len(to_remove)} jobs from results_store.")

'''
BIRD SOUND 22
- Bird sound recognition using transformers pipeline for audio classification.
'''

def process_audio(job_id, file_path):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_path)[1], dir='/filedata/AiModel/tmp')
    temp_file.close()
    shutil.copyfile(file_path, temp_file.name)

    try:
        if file_path.endswith('.mp3'):
            audio = AudioSegment.from_mp3(temp_file.name)
            wav_path = temp_file.name.replace('.mp3', '.wav')
            audio.export(wav_path, format="wav")
            temp_file.name = wav_path

        audio = AudioSegment.from_wav(temp_file.name)
        audio_duration = len(audio) / 1000

        spectrogram_path = None

        if audio_duration <= 300:
            y, sr = librosa.load(temp_file.name, sr=None)
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(librosa.amplitude_to_db(librosa.stft(y), ref=np.max), y_axis=None, x_axis=None)
            plt.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            spectrogram_path = f'/filedata/AiModel/sound_reco_output/spectrogram_{job_id}.png'
            plt.savefig(spectrogram_path, bbox_inches='tight', pad_inches=0)
            plt.close()

        segment_length = 2
        segments = [audio[i:i + segment_length * 1000] for i in range(0, len(audio), segment_length * 1000)]

        if len(segments[-1]) < 1000 and len(segments) > 1:
            segments.pop()

        start_time = time.time()
        formatted_result = OrderedDict({
            "time_window": segment_length,
            "total_segments": len(segments),
            "duration": 0,
            "results": [],
            "spectrogram": spectrogram_path
        })

        bird_segments = []
        noise_segments = []
        current_bird_start = None
        current_bird_classes = set()
        current_noise_start = None

        for i, segment in enumerate(segments):
            if results_store[job_id].get('cancelled'):
                results_store[job_id]['status'] = 'cancelled'
                return

            segment_file = f"/filedata/AiModel/tmp/segment_{i}.wav"
            segment.export(segment_file, format="wav")

            # 模型推理
            result = pipe(segment_file)
            bird_classes = [item['label'] for item in result if item['label'] != '噪声' and item['score'] > 0.1]
            noise_present = all(item['label'] == '噪声' for item in result if item['score'] > 0.1)

            # 如果前一个时间窗口预测结果和当前一致，合并时间窗口并取平均置信度
            if formatted_result['results'] and bird_classes:
                prev_result = formatted_result['results'][-1]
                prev_classes = prev_result['classes']

                if set(bird_classes) == set(prev_classes):
                    # 更新时间窗口
                    prev_result['end_time'] = (i + 1) * segment_length
                    # 更新置信度平均值
                    prev_result['scores'] = [
                        (prev_score + current_score) / 2
                        for prev_score, current_score in zip(prev_result['scores'], [item['score'] for item in result if item['score'] > 0.1])
                    ]
                    continue

            # 否则，正常处理并加入到结果
            formatted_result['results'].append(
                OrderedDict({
                    "segment_id": i,
                    "start_time": i * segment_length,
                    "end_time": (i + 1) * segment_length,
                    "classes": bird_classes,
                    "labels": [i for i, item in enumerate(result) if item['score'] > 0.1],
                    "scores": [item['score'] for item in result if item['score'] > 0.1]
                })
            )

            chunk_end_time = time.time()
            progress = (i + 1) / len(segments)
            results_store[job_id]['progress'] = f'{i+1}/{len(segments)}'
            results_store[job_id]['estimated_remaining_time'] = estimate_remaining_time(start_time, chunk_end_time, progress)

        # 处理最后一个未完成的鸟类或噪声段
        if current_bird_start is not None:
            bird_segments.append({
                "start_time": current_bird_start,
                "end_time": len(segments) * segment_length,
                "bird_classes": list(current_bird_classes)
            })

        if current_noise_start is not None:
            noise_segments.append({
                "start_time": current_noise_start,
                "end_time": len(segments) * segment_length,
                "noise_label": "噪声"
            })

        # 生成 Excel 文件
        excel_file_path = f"/filedata/AiModel/sound_reco_output/report_{job_id}.xlsx"
        workbook = xlsxwriter.Workbook(excel_file_path)
        worksheet = workbook.add_worksheet()

        worksheet.write(0, 0, "Start Time (s)")
        worksheet.write(0, 1, "End Time (s)")
        worksheet.write(0, 2, "Label")

        for idx, segment in enumerate(bird_segments, start=1):
            worksheet.write(idx, 0, segment["start_time"])
            worksheet.write(idx, 1, segment["end_time"])
            worksheet.write(idx, 2, ", ".join(segment["bird_classes"]))

        for idx, noise_segment in enumerate(noise_segments, start=len(bird_segments) + 1):
            worksheet.write(idx, 0, noise_segment["start_time"])
            worksheet.write(idx, 1, noise_segment["end_time"])
            worksheet.write(idx, 2, noise_segment["noise_label"])

        workbook.close()

        end_time = time.time()
        formatted_result['duration'] = end_time - start_time

        results_store[job_id] = {
            'status': 'completed',
            'start_time': datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S'),
            'origin_filepath': file_path,
            'result': formatted_result,
            'excel_report': os.path.abspath(excel_file_path)
        }

        try:
            response = requests.post(callback_url, json={
                "jobId": job_id,
                "recognizeType": 2,
                "recognizeResult": results_store[job_id]
            })
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Callback failed: {e}")

    except Exception as e:
        results_store[job_id] = {'status': 'failed', 'error': str(e)}
    finally:
        os.remove(temp_file.name)

# 工作线程函数，处理队列中的任务
def sound_worker():
    while True:
        job_id, file_path = sound_queue.get()
        results_store[job_id] = {'status': 'processing', 'cancelled': False, 'progress': '0/0', 'estimated_remaining_time': ''}
        process_audio(job_id, file_path)
        sound_queue.task_done()

# 启动工作线程
threading.Thread(target=sound_worker, daemon=True).start()

# 音频识别接口
@app.route('/bird22_sound_recognition_torch', methods=['POST'])
def bird15_sound_recognition():
    data = request.get_json()
    if not data or 'file_path' not in data:
        return jsonify({'error': 'No file path provided'}), 400, {'Content-Type': 'application/json'}

    file_path = data['file_path']

    if not os.path.exists(file_path):
        return jsonify({'error': 'File does not exist'}), 400, {'Content-Type': 'application/json'}

    job_id = str(uuid.uuid4())
    results_store[job_id] = {'status': 'queued', 'cancelled': False}
    sound_queue.put((job_id, file_path))

    return jsonify({'job_id': job_id, 'status': 'queued'}), 200, {'Content-Type': 'application/json'}


'''
BIRD IMAGE 100
- Bird image detection using mmdetection.
'''

def process_image(job_id, file_path):
    try:
        start_time = time.time()
        results = []

        for i, path in enumerate(file_path):
            # 检查任务是否被取消
            if results_store[job_id].get('cancelled'):
                results_store[job_id]['status'] = 'cancelled'
                return

            # 读取图像
            img = cv2.imdecode(np.frombuffer(open(path, 'rb').read(), np.uint8), cv2.IMREAD_COLOR)

            result = yolov11_model.predict(source=img,
                                           imgsz=1536,
                                            conf=0.3,
                                            save=False,
                                            show=False,
                                            verbose=False,
                                            )

            bboxes = result[0].boxes.xyxy.cpu().numpy().tolist()
            labels = list(map(int, result[0].boxes.cls.cpu().numpy().tolist()))
            scores = result[0].boxes.conf.cpu().numpy().tolist()
            classes = [bird100_categories[label] for label in labels]
            palette = [bird100_palette[label] for label in labels]

            formatted_result = {
                'bboxes': bboxes,
                'classes': classes,
                'palette': palette,
                'labels': labels,
                'scores': scores
            }
            results.append(formatted_result)

            chunk_end_time = time.time()

            # 每处理一个chunk，更新状态和预计时间
            progress = (i+1) / len(file_path)
            results_store[job_id]['progress'] = f'{i+1}/{len(file_path)}'
            results_store[job_id]['estimated_remaining_time'] = estimate_remaining_time(start_time, chunk_end_time, progress)

        end_time = time.time()

        # 更新任务状态为 completed
        results_store[job_id] = {
            'status': 'completed',
            'estimated_processing_time': end_time - start_time,
            'start_time': datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S'),
            'origin_filepath': file_path if len(file_path) > 1 else file_path[0],
            'result': results if len(file_path) > 1 else results[0]
        }

        # 添加回调请求
        try:
            response = requests.post(callback_url, json={
                "jobId": job_id,
                "recognizeType": 1,
                "recognizeResult": results_store[job_id]
            })
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Callback failed: {e}")

    except Exception as e:
        results_store[job_id] = {'status': 'failed', 'error': str(e)}


# 工作线程函数，处理队列中的任务
def detection_worker():
    while True:
        job_id, file_path = detection_queue.get()  # 从队列中获取任务
        results_store[job_id] = {'status': 'processing', 'cancelled': False, 'progress': '0/0', 'estimated_remaining_time': ''}  # 更新任务状态为 processing
        process_image(job_id, file_path)  # 处理图像任务
        detection_queue.task_done()  # 任务完成

# 启动工作线程
threading.Thread(target=detection_worker, daemon=True).start()

# 接口：提交图像检测任务
@app.route('/bird100_detection_torch', methods=['POST'])
def bird100_detection():
    data = request.json
    if not data or 'file_path' not in data:
        return jsonify({'error': 'No file path provided'}), 400, {'Content-Type': 'application/json'}

    file_path = data['file_path'].split('+')
    for path in file_path:
        if not os.path.exists(path):
            return jsonify({'error': 'File not found'}), 400, {'Content-Type': 'application/json'}

    job_id = str(uuid.uuid4())  # 生成唯一的任务ID
    results_store[job_id] = {'status': 'queued', 'cancelled': False}  # 初始化任务状态为 queued
    detection_queue.put((job_id, file_path))  # 将任务加入队列

    return jsonify({'job_id': job_id, 'status': 'queued'}), {'Content-Type': 'application/json'}


'''
BIRD VIDEO 100
'''

# 元数据处理函数
def process_video_meta(job_id, file_path):
    try:
        video = cv2.VideoCapture(file_path)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        start_time = time.time()
        results = []

        while True:
            if results_store[job_id].get('cancelled'):
                results_store[job_id]['status'] = 'cancelled'
                return

            ret, frame = video.read()
            if not ret:
                break

            result = yolov11_model.predict(source=frame,
                                           imgsz=1536,
                                            conf=0.3,
                                            save=False,
                                            show=False,
                                            verbose=False,
                                            )

            bboxes = result[0].boxes.xyxy.cpu().numpy().tolist()
            labels = list(map(int, result[0].boxes.cls.cpu().numpy().tolist()))
            scores = result[0].boxes.conf.cpu().numpy().tolist()
            classes = [bird100_categories[label] for label in labels]
            palette = [bird100_palette[label] for label in labels]

            formatted_result = {
                'bboxes': bboxes,
                'labels': labels,
                'scores': scores,
                'classes': classes,
                'palettes': palette
            }

            results.append(formatted_result)

            chunk_end_time = time.time()

            # 每处理一个chunk，更新状态和预计时间
            progress = (frame_count + 1) / total_frames
            results_store[job_id]['progress'] = f'{frame_count+1}/{total_frames}'
            results_store[job_id]['estimated_remaining_time'] = estimate_remaining_time(start_time, chunk_end_time, progress)

            frame_count += 1

        video.release()
        end_time = time.time()
        inference_duration = end_time - start_time
        inference_fps = round(frame_count / inference_duration, 0)

        # 保存结果
        results_store[job_id] = {
            'status': 'completed',
            'duration': inference_duration,
            'fps': inference_fps,
            'origin_filepath': file_path,
            'start_time': datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S'),
            'results': results,
        }

        # 添加回调请求
        try:
            response = requests.post(callback_url, json={
                "jobId": job_id,
                "recognizeType": 3,
                "recognizeResult": results_store[job_id]
            })
            response.raise_for_status()  # 确保请求成功
        except requests.RequestException as e:
            print(f"Callback failed: {e}")

    except Exception as e:
        results_store[job_id] = {'status': 'failed', 'error': str(e)}

# 视频元数据 API
@app.route('/bird100_detection_video_meta', methods=['POST'])
def bird100_detection_video_meta():
    data = request.get_json()
    file_path = data.get('file_path')

    if not file_path or not os.path.exists(file_path):
        return jsonify({'error': 'Invalid file path'}), 400, {'Content-Type': 'application/json'}

    job_id = str(uuid.uuid4())  # 生成唯一的任务ID
    results_store[job_id] = {'status': 'queued', 'cancelled': False}  # 初始化任务状态为 queued
    video_queue.put((job_id, file_path, 'meta'))  # 将元数据任务加入队列

    return jsonify({'job_id': job_id, 'status': 'queued'}), {'Content-Type': 'application/json'}


# 任务处理函数
def process_video(job_id, file_path):
    try:
        # 生成带年/月分级目录的输出路径
        current_time = datetime.now()
        year_month_path = current_time.strftime('%Y/%m')
        output_dir = os.path.join('/filedata/AiModel/image_reco_output', year_month_path)

        # 确保目标目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 临时文件和输出文件路径
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4', dir='/filedata/AiModel/tmp')
        temp_video_file = os.path.join(output_dir, f"{job_id}_processed.mp4")
        shutil.copy(file_path, temp_file.name)  # 复制文件到临时文件

        # 以下代码保持不变
        video = cv2.VideoCapture(temp_file.name)
        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # 处理输出流
        out = imageio.get_writer(temp_video_file, format='ffmpeg', mode='I', fps=fps, codec='libx264', quality=5)

        frame_count = 0
        start_time = time.time()

        results = []

        while True:
            if results_store[job_id].get('cancelled'):
                results_store[job_id]['status'] = 'cancelled'
                return
            # 总共的帧数
            ret, frame = video.read()
            if not ret:
                break

            result = yolov11_model.predict(source=frame,
                                           imgsz=1536,
                                            conf=0.3,
                                            save=False,
                                            show=False,
                                            verbose=False,
                                            )

            bboxes = result[0].boxes.xyxy.cpu().numpy().tolist()
            labels = list(map(int, result[0].boxes.cls.cpu().numpy().tolist()))
            scores = result[0].boxes.conf.cpu().numpy().tolist()
            classes = [bird100_categories[label] for label in labels]
            palette = [bird100_palette[label] for label in labels]

            is_show_class_99 = sum([1 for label in labels if not label == 99]) == 0

            frame_results = {
                'bboxes': bboxes,
                'labels': labels,
                'scores': scores,
                'classes': classes,
                'palettes': palette
            }

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 绘制框和标签
            for bbox, label, score, cls_item, color in zip(bboxes, labels, scores, classes, palette):
                if not is_show_class_99:
                    if label == 99:
                        continue

                bbox = list(map(int, bbox))

                text = f'{cls_item} {score:.2f}'
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_width, text_height = text_size
                cv2.rectangle(frame, (bbox[0] - 1, bbox[1] - int(text_height * 1.5) - 2), (bbox[0] + int(text_width * 1.5), bbox[1]), color, -1)
                cv2.putText(frame, text, (bbox[0], bbox[1] - 10), (255, 255, 255), cv2.FontFace("UTF8"), 40)
                frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

            results.append(frame_results)
            out.append_data(frame)

            chunk_end_time = time.time()

            # 每处理一个chunk，更新状态和预计时间
            progress = (frame_count + 1) / total_frames
            results_store[job_id]['progress'] = f'{frame_count+1}/{total_frames}'
            results_store[job_id]['estimated_remaining_time'] = estimate_remaining_time(start_time, chunk_end_time, progress)

            frame_count += 1

        end_time = time.time()
        inference_duration = end_time - start_time
        inference_fps = round(frame_count / inference_duration, 0)

        video.release()
        out.close()
        os.remove(temp_file.name)

        # 更新任务状态为 completed
        results_store[job_id] = {
            'status': 'completed',
            'duration': inference_duration,
            'fps': inference_fps,
            'video_path': temp_video_file,
            'origin_filepath': file_path,
            'start_time': datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S'),
            'results': results
        }

        # 添加回调请求
        try:
            response = requests.post(callback_url, json={
                "jobId": job_id,
                "recognizeType": 3,
                "recognizeResult": results_store[job_id]
            })
            response.raise_for_status()  # 确保请求成功
        except requests.RequestException as e:
            print(f"Callback failed: {e}")

    except Exception as e:
        results_store[job_id] = {'status': 'failed', 'error': str(e)}

# 工作线程函数，处理队列中的任务
def video_worker():
    while True:
        job_id, file_path, task_type = video_queue.get()  # 获取任务和任务类型
        results_store[job_id] = {'status': 'processing', 'cancelled': False, 'progress': '0/0', 'estimated_remaining_time': ''}

        if task_type == 'detection':
            process_video(job_id, file_path)  # 执行检测任务
        elif task_type == 'meta':
            process_video_meta(job_id, file_path)  # 执行元数据任务

        video_queue.task_done()  # 任务完成

# 启动工作线程
threading.Thread(target=video_worker, daemon=True).start()

@app.route('/bird100_detection_video', methods=['POST'])
def bird100_detection_video():
    data = request.get_json()
    if not data or 'file_path' not in data:
        return jsonify({'error': 'No file path provided'}), 400, {'Content-Type': 'application/json'}

    file_path = data['file_path']
    if not os.path.exists(file_path):
        return jsonify({'error': 'Invalid file path'}), 400, {'Content-Type': 'application/json'}

    job_id = str(uuid.uuid4())  # 生成唯一的任务ID
    results_store[job_id] = {'status': 'queued', 'cancelled': False}  # 初始化任务状态为 queued
    video_queue.put((job_id, file_path, 'detection'))  # 将检测任务加入队列

    return jsonify({'job_id': job_id, 'status': 'queued'}), {'Content-Type': 'application/json'}


'''
BIRD STREAM 100
'''

class VideoStream:
    def __init__(self, stream_url, stream_id, model, num_workers=8):
        self.stream_url = stream_url
        self.stream_id = stream_id
        self.model = model
        self.is_streaming = True
        self.lock = threading.Lock()
        self.retry_attempts = 6
        self.num_workers = num_workers
        self.frame_queue = Queue(maxsize=10)

        # Initialize the video capture stream
        self._initialize_capture()

        # FFmpeg command to push the stream
        rtmp_address = f'rtmp://0.0.0.0:1935/tmp/{stream_id}'
        self._initialize_ffmpeg(rtmp_address)

        # Thread to capture frames
        self.read_thread = threading.Thread(target=self._read_frames)
        self.read_thread.daemon = True
        self.read_thread.start()

        # ThreadPoolExecutor for parallel frame processing
        self.executor = ThreadPoolExecutor(max_workers=num_workers)

        # Thread to process frames
        self.process_thread = threading.Thread(target=self._process_frames)
        self.process_thread.daemon = True
        self.process_thread.start()

    def _initialize_capture(self):
        """ Initialize video capture with retry mechanism """
        self.cap = cv2.VideoCapture(self.stream_url)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.fps = 20
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

    def _initialize_ffmpeg(self, rtmp_address):
        """ Initialize FFmpeg command and pipe """
        self.ffmpeg_command = [
            'ffmpeg',
            '-y',
            '-re',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-acodec', 'aac',
            '-s', '1280x720',
            '-r', f'{self.fps}',
            '-i', '-',
            '-c:v', 'h264_nvenc',
            '-pix_fmt', 'yuv420p',
            '-preset', 'fast',
            '-vprofile', 'baseline',
            '-cq', '19',
            '-b:v', '5000k',
            '-maxrate', '5000k',
            '-bufsize', '6000k',
            '-g', '30',
            '-f', 'flv',
            rtmp_address
        ]
        self._start_ffmpeg()

    def _start_ffmpeg(self):
        """Start the FFmpeg pipe with error handling"""
        try:
            self.ffmpeg_pipe = sp.Popen(self.ffmpeg_command, stdin=sp.PIPE)
        except Exception as e:
            print(f"Error initializing FFmpeg: {e}")
            self.release()

    def release(self):
        """ Ensure the stream is closed properly. """
        with self.lock:
            if not self.is_streaming:
                return

            self.is_streaming = False

            if self.read_thread and self.read_thread.is_alive():
                self.read_thread.join(timeout=2)

            if self.process_thread and self.process_thread.is_alive():
                self.process_thread.join(timeout=2)

            self.executor.shutdown(wait=True)

            if self.ffmpeg_pipe.stdin:
                self.ffmpeg_pipe.stdin.close()

            self.ffmpeg_pipe.wait()

            print(f"Stream {self.stream_url} stopped.")

    def _read_frames(self):
        """ Read frames from the video stream using OpenCV and handle retries by reinitializing the capture if it fails. """
        retry_count = 0
        while self.is_streaming:
            try:
                # Initialize the capture again in case it's the first attempt or after a failure
                self._initialize_capture()

                if not self.cap.isOpened():
                    raise Exception(f"Unable to open stream {self.stream_url}")

                while self.is_streaming:
                    ret, frame = self.cap.read()
                    if not ret:
                        print(f"Failed to read frame from stream {self.stream_url}. Retrying...")
                        retry_count += 1
                        self.cap.release()
                        time.sleep(1)  # Small delay before retrying
                        self._initialize_capture()  # Reinitialize capture after failure
                        continue

                    retry_count = 0  # Reset retry count on successful frame read

                    try:
                        self.frame_queue.put(frame, timeout=2)
                    except queue.Full:
                        print("Frame queue is full, dropping frame.")

                # Release capture once the loop ends or if streaming stops
                self.cap.release()

            except Exception as e:
                print(f"Error reading stream {self.stream_url}: {e}")
                retry_count += 1
                if retry_count > self.retry_attempts:
                    print(f"Failed to restart stream after {self.retry_attempts} attempts.")
                    self.release()  # Stop the stream if max retries are exceeded
                    break
                time.sleep(1)

            finally:
                if self.cap:
                    self.cap.release()

    def _process_frames(self):
        """Retrieve frames from the queue and process them."""
        while self.is_streaming:
            try:
                frame_batch = [self.frame_queue.get(timeout=2)]  # Batch of 1 for real-time
                if frame_batch:
                    self.executor.submit(self._process_batch, frame_batch)
            except queue.Empty:
                print("Frame queue is empty.")

    def _process_batch(self, frame_batch):
        """ Batch process frames and push to the
        stream. """
        try:
            frame = frame_batch[0]
            result = yolov11_model.predict(source=frame,
                                           imgsz=1536,
                                            conf=0.3,
                                            save=False,
                                            show=False,
                                            verbose=False,
                                            )

            bboxes = result[0].boxes.xyxy.cpu().numpy().tolist()
            labels = list(map(int, result[0].boxes.cls.cpu().numpy().tolist()))
            scores = result[0].boxes.conf.cpu().numpy().tolist()
            classes = [bird100_categories[label] for label in labels]
            palette = [bird100_palette[label] for label in labels]

            is_show_class_99 = sum([1 for label in labels if not label == 99]) == 0

            frame = self._draw_bboxes(frame, bboxes, labels, scores, classes, palette, is_show_class_99)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
            frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
            frame = cv2.resize(frame, (1280, 720))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
            self.ffmpeg_pipe.stdin.write(frame.tobytes())

        except Exception as e:
            print(f"Error processing frame batch: {e}")

    def _draw_bboxes(self, frame, bboxes, labels, scores, classes, palette, is_show_class_99):
        # 绘制框和标签
        for bbox, label, score, cls_item, color in zip(bboxes, labels, scores, classes, palette):
            if not is_show_class_99:
                if label == 99:
                    continue

            bbox = list(map(int, bbox))

            text = f'{cls_item} {score:.2f}'
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_width, text_height = text_size
            cv2.rectangle(frame, (bbox[0] - 1, bbox[1] - int(text_height * 1.5) - 2), (bbox[0] + int(text_width * 1.5), bbox[1]), color, -1)
            cv2.putText(frame, text, (bbox[0], bbox[1] - 10), (255, 255, 255), cv2.FontFace("UTF8"), 40)
            frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        return frame

@app.route('/start_stream', methods=['POST'])
def start_or_fetch_stream():
    """启动或获取视频流，返回推流地址和所有流"""
    data = request.json
    stream_url = data.get('stream_url')
    if not stream_url:
        return jsonify({'error': 'No stream URL provided'}), 400, {'Content-Type': 'application/json'}

    index = {
        'rtmp://1.94.51.225:1935/rtp/32011401091187000002_32041101091327000001': 1,
        'rtmp://1.94.51.225:1935/rtp/32011401091187000002_32041101091327000002': 2,
        'rtmp://1.94.51.225:1935/rtp/32011401091187000002_32041101091327000003': 3,
        'rtmp://1.94.51.225:1935/rtp/32011401091187000002_32041101091327000004': 4,
        'rtmp://1.94.51.225:1935/rtp/32011401091187000002_32041101091327000005': 5,
        'rtmp://1.94.51.225:1935/rtp/32011401091187000002_32041101091327000006': 6,
        'rtmp://liteavapp.qcloud.com/live/liteavdemoplayerstreamid': 7,
    }
    # stream_hash = hashlib.md5(stream_url.encode()).hexdigest()
    stream_hash = index[stream_url]

    # 检查流是否已经启动
    if stream_hash in video_streams:
        rtmp_address = f'rtmp://180.101.130.45:1935/tmp/{stream_hash}'
        all_stream_addresses = [f'rtmp://180.101.130.45:1935/tmp/{hash_key}' for hash_key in video_streams.keys()]
        return jsonify({
            'status': 'Stream already started',
            'rtmp_address': rtmp_address,
            'all_stream_addresses': all_stream_addresses
        }), 200, {'Content-Type': 'application/json'}

    # 如果推流数超过2，则释放最早的一个流
    if len(video_streams) >= 1:
        stop_stream()

    # 启动新的视频流
    video_streams[stream_hash] = VideoStream(stream_url, stream_hash, model)
    rtmp_address = f'rtmp://180.101.130.45:1935/tmp/{stream_hash}'
    all_stream_addresses = [f'rtmp://180.101.130.45:1935/tmp/{hash_key}' for hash_key in video_streams.keys()]

    return jsonify({
        'status': 'Stream started',
        'rtmp_address': rtmp_address,
        'all_stream_addresses': all_stream_addresses
    }), 200, {'Content-Type': 'application/json'}


@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    # 将所有要删除的stream hash存储起来
    streams_to_stop = list(video_streams.keys())

    for stream_hash in streams_to_stop:
        stream = video_streams[stream_hash]
        stream.release()
        del video_streams[stream_hash]

    return jsonify({'status': 'All streams stopped'}), 200, {'Content-Type': 'application/json'}

if __name__ == '__main__':
    threading.Thread(target=check_heartbeats, daemon=True).start()
    server = pywsgi.WSGIServer(('0.0.0.0', 5000), app)
    server.serve_forever()
    while True:
        check_and_clear_results_store()
        time.sleep(0.05)