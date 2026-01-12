# 将MultiPolygon的geojson文件转MultiLineString的geojson文件
import json
from shapely.geometry import shape, mapping, MultiLineString, MultiPolygon


def convert_multipolygon_to_multilinestring(input_geojson, output_geojson):
    # 读取GeoJSON文件
    with open(input_geojson, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 转换每个MultiPolygon为MultiLineString
    for feature in data['features']:
        geom = shape(feature['geometry'])
        if isinstance(geom, MultiPolygon):  # 检查是否为MultiPolygon
            # 提取所有的外环线
            lines = [polygon.exterior for polygon in geom.geoms]
            multilinestring = MultiLineString(lines)
            # 替换geometry为MultiLineString
            feature['geometry'] = mapping(multilinestring)

    # 将结果写回新的GeoJSON文件
    with open(output_geojson, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# 输入文件和输出文件路径
input_file = 'E:/南山风景区/data/1749019443406/nanan.json'
output_file = 'E:/南山风景区/jeecgboot-vue3/src/assets/largeScreen/map/nanan_multilinestring.json'

convert_multipolygon_to_multilinestring(input_file, output_file)
print(f"转换完成，结果已保存到 {output_file}")
