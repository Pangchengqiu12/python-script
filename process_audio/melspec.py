import sys
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
import tempfile
import os
import shutil
import json
import time
result = {
    "status":0,
    "data":[],
    "message":"",
    'timestamp':0
}
def load_audio_segment(file_path):
 # 获取文件的扩展名（不包含点）
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.mp3':
        audio = AudioSegment.from_mp3(file_path)
    elif ext == '.wav':
        audio = AudioSegment.from_wav(file_path)
    else:
        raise ValueError("Unsupported file format. Only .mp3 and .wav are supported.")

    return audio
# 生成语谱图
def generate_mel_spectrogram(audio_path, output_path):
    try:
        # 读取音频文件
        audio_data, sample_rate = librosa.load(audio_path, sr=None)

        # 计算 Mel 频率谱图
        S = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)

        # 显示并保存语谱图
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_dB, sr=sample_rate, x_axis='time', y_axis='mel', fmax=8000)
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel-frequency spectrogram')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        return "SUCCESS"  # 返回状态
    except Exception as e:
        result["status"] = 0
        result["message"] = f"ERROR in generate_mel_spectrogram: {e}"
        print(json.dumps(result, ensure_ascii=False, indent=4))
        return "FAIL"
# 处理音频
def process_audio(file_path, save_path):
    temp_dir = os.path.join(tempfile.gettempdir(),'process_audio')  # 音频临时存储文件夹
    # 如果临时文件夹不存在，则创建
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    file_name_without_extension = os.path.splitext(os.path.basename(file_path))[0] # 获取文件名（不包含扩展名）

    # 如果保存路径不存在，则创建
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    try:
        # 创建临时文件并将原始文件复制到临时文件
        # temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_path)[1], dir=temp_dir)
        # temp_file.close()  # 关闭文件句柄，但不删除文件
        # shutil.copyfile(file_path, temp_file.name)

        # if file_path.endswith('.mp3'):
        #     audio = AudioSegment.from_mp3(temp_file.name)
        #     wav_path = temp_file.name.replace('.mp3', '.wav')
        #     audio.export(wav_path, format="wav")
        #     os.remove(temp_file.name)
        #     temp_file.name = wav_path

        audio = load_audio_segment(file_path)

        audio_duration = len(audio) / 1000

        segment_length = 10
        segments = [audio[i:i + segment_length * 1000] for i in range(0, len(audio), segment_length * 1000)]
        if len(segments[-1]) < 1000 and len(segments) > 1:
            segments.pop()
        process_data = []  # 最后的结果
        for i, segment in enumerate(segments):
            segment_file = os.path.join(save_path,f"{file_name_without_extension}_{i}{os.path.splitext(file_path)[1]}")
            spectrogram_path =os.path.join(save_path,f"{file_name_without_extension}_{i}.jpg")
            segment.export(segment_file, format="wav")
            status = generate_mel_spectrogram(segment_file, spectrogram_path)
            if status == "SUCCESS":
                process_data.append({
                    "segment_file": segment_file,
                    "spectrogram_path": spectrogram_path
                })
            else:
                print(f"Failed to generate mel spectrogram for segment {i}")

    except Exception as e:
        result["status"] = 0
        result["message"] = f"ERROR in generate_mel_spectrogram: {e}"
        print(json.dumps(result, ensure_ascii=False, indent=4))
        return result
    finally:
        # os.remove(temp_file.name)
        return process_data

if __name__ == "__main__":
    # 检查参数长度是不是等于3，sys.argv[0] 是 脚本本身的文件名（例如 melspec.py）
    # if len(sys.argv) != 3:
    #     print("ERROR: Invalid arguments")
    # else:
    #     audio_path = sys.argv[1] # 音频地址
    #     output_path = sys.argv[2] # 输出文件夹地址
    #     result["timestamp"] = time.time()
    #     for i, path in enumerate(json.loads(audio_path)):
    #         data = process_audio(path, output_path)
    #         result["data"].append({"id":i,"audio_path":path,"data":data})



    audio_path = [
        'E:/mine/python/python-script/process_audio/104.wav',
        'E:/assets/音频/$RJZ75H1.mp3',
        'E:/assets/音频/$RJZ75H1(1).mp3',
        'E:/assets/音频/$RJZ75H1(2).mp3',
        'E:/assets/音频/$RJZ75H1(3).mp3',
        'E:/assets/音频/$RJZ75H1(4).mp3'
      ]
    output_path = 'E:/demo'
    result["timestamp"] = time.time()
    for i, path in enumerate(audio_path):
        data = process_audio(path, output_path)
        result["data"].append({"id":i,"audio_path":path,"data":data})

    # 将Python对象转换为JSON字符串
    result["timestamp"] = int((time.time() - result["timestamp"]) * 1000)
    result["status"] = 1
    json_data = json.dumps(result, ensure_ascii=False, indent=4)
    print(json_data)
