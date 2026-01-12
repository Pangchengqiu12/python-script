import pandas as pd
import requests
import time
import ssl
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

# ==================== 配置区域 ====================
# Excel 文件路径（修改为你的实际路径）
EXCEL_FILE = "E:\固原\data\固原鸣声采样点.xlsx"  # 如 r'D:\data\equipment.xlsx'

# 接口信息
URL = "http://202.100.121.22:9000/guyuanbio/obVoiceEquipment/add"

# 请求头（注意：如果 token 已过期，请重新登录系统获取新 token 替换）
HEADERS = {
    "X-Access-Token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjE3NjgwNjc0ODAsInVzZXJuYW1lIjoiZ3V5dWFuIn0.-eJxl9_0xt6WLDickNpDs8sDtAMkEJV2sXuatAlbp4M",
    "User-Agent": "Apifox/1.0.0[](https://apifox.com)",
    "Content-Type": "application/json",
    "Accept": "*/*",
}

# 固定字段
GROUP_CODE = "2009510671527809026"

# Excel 列映射（A=0, B=1, C=2, ... 因为 pandas 用 0-based 索引）
COLUMN_MAPPING = {
    "sncode": 0,  # A列 → 第0列
    "latitude": 1,  # B列 → 第1列
    "longitude": 2,  # C列 → 第2列
    "location": 7,  # H列 → 第7列
    "equipmentName": 9,  # J列 → 第9列
}

# 是否跳过 Excel 第一行（标题行）
SKIP_HEADER = True

# 每次请求间隔（秒），避免太快被限流
DELAY = 0.5

# 是否忽略 SSL 证书验证（目标服务器可能是自签名证书，常见于内网）
VERIFY_SSL = False
# ================================================


def main():
    # 读取 Excel
    try:
        if SKIP_HEADER:
            df = pd.read_excel(EXCEL_FILE, header=None)
        else:
            df = pd.read_excel(EXCEL_FILE)
        print(f"成功读取 Excel，共 {len(df)} 行数据")
    except Exception as e:
        print(f"读取 Excel 失败: {e}")
        return

    # 创建 requests Session，提升性能并支持连接复用
    session = requests.Session()

    # 可选：设置重试机制（网络不稳定时有用）
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # 忽略 SSL 警告（如果 VERIFY_SSL=False）
    if not VERIFY_SSL:
        requests.packages.urllib3.disable_warnings()

    success_count = 0
    fail_count = 0

    # 从第几行开始处理（跳过标题行）
    start_row = 1 if SKIP_HEADER else 0

    for idx in range(start_row, len(df)):
        row = df.iloc[idx]
        row_num = idx + 1  # Excel 实际行号

        try:
            sncode = str(row[COLUMN_MAPPING["sncode"]]).strip()
            equipmentName = str(row[COLUMN_MAPPING["equipmentName"]]).strip()
            location = str(row[COLUMN_MAPPING["location"]]).strip()
            longitude = float(row[COLUMN_MAPPING["longitude"]])
            latitude = float(row[COLUMN_MAPPING["latitude"]])

            if not sncode or sncode == "nan":
                print(f"第 {row_num} 行：sncode 为空，跳过")
                continue

            # 构造请求数据
            payload = {
                "sncode": sncode,
                "equipmentName": equipmentName,
                "location": location,
                "longitude": longitude,
                "latitude": latitude,
                "groupCode": GROUP_CODE,
            }

            # 发送 POST 请求
            response = session.post(
                URL,
                json=payload,  # 自动序列化为 JSON 并设置正确编码
                headers=HEADERS,
                verify=VERIFY_SSL,  # False 表示忽略证书验证
                timeout=30,
            )

            if response.status_code in (200, 201):
                print(f"第 {row_num} 行上传成功：{sncode}")
                success_count += 1
            else:
                print(
                    f"第 {row_num} 行上传失败：{sncode} | 状态码: {response.status_code} | 响应: {response.text}"
                )
                fail_count += 1

        except Exception as e:
            print(f"第 {row_num} 行处理异常：{e}")
            fail_count += 1

        time.sleep(DELAY)

    session.close()
    print("\n" + "=" * 50)
    print(f"上传完成！成功：{success_count} 条，失败：{fail_count} 条")
    print("=" * 50)


if __name__ == "__main__":
    main()
