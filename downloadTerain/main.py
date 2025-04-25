import os
import json
import requests
from tqdm import tqdm

headers={
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
    "Referer": "http://studio.mars3d.cn/",
    "Origin": "http://studio.mars3d.cn"
}

# 配置项
LAYER_JSON_PATH = "json/layer.json"  # 本地 layer.json 路径
TERRAIN_URL_ROOT = "https://data1.mars3d.cn/terrain"  # 替换为你的地形服务地址
OUTPUT_DIR = "./downloaded_terrain"  # 下载保存路径

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 加载 layer.json
with open(LAYER_JSON_PATH, "r", encoding="utf-8") as f:
    layer_data = json.load(f)

tiles_template = layer_data["tiles"][0]  # 例如 "{z}/{x}/{y}.terrain"
available = layer_data["available"]

# 遍历每一层的 available 坐标块
for z, ranges in enumerate(available):
    for block in ranges:
        start_x = block["startX"]
        end_x = block["endX"]
        start_y = block["startY"]
        end_y = block["endY"]

        # 循环 tile 范围
        for x in range(start_x, end_x + 1):
            for y in range(start_y, end_y + 1):
                # 构造 URL 和保存路径
                tile_path = tiles_template.replace("{z}", str(z)).replace("{x}", str(x)).replace("{y}", str(y))
                tile_url = f"{TERRAIN_URL_ROOT}/{tile_path}"
                save_path = os.path.join(OUTPUT_DIR, str(z), str(x))
                os.makedirs(save_path, exist_ok=True)

                filename = os.path.join(save_path, f"{y}.terrain")

                # 下载文件
                try:
                    resp = requests.get(tile_url,headers=headers, timeout=10)
                    resp.raise_for_status()
                    with open(filename, "wb") as f:
                        f.write(resp.content)
                    print(f"✅ 下载成功: {tile_url}")
                except Exception as e:
                    print(f"❌ 下载失败: {tile_url} 原因: {e}")
