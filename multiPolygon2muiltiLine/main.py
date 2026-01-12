import geopandas as gpd

# 读取 GeoJSON 文件
gdf = gpd.read_file("your_polygon.geojson")

# 检查几何类型
print(gdf.geom_type.unique())  # 应该是 Polygon 或 MultiPolygon

# 提取边界（转换为线）
gdf["boundary"] = gdf.geometry.boundary

# 构建新的 GeoDataFrame，只保留边界线
lines_gdf = gpd.GeoDataFrame(gdf[["boundary"]], geometry="boundary", crs=gdf.crs)

# 保存为新的 GeoJSON 文件
lines_gdf.to_file("output_lines.geojson", driver="GeoJSON")