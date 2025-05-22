import numpy as np
import pandas as pd

# 地球半径 (km) 和 轨道高度
R = 6371  # 地球平均半径
h = 550  # 轨道高度
r = R + h  # 轨道半径

# 卫星地理坐标 (纬度, 经度)
satellites = [
    (59, 100), (60, 130), (61, 160),
    (38.5, 160), (37.5, 130), (36.5, 100),
    (14, 100), (15, 130), (16, 160),
]

# 计算极坐标 (x, y, z)
satellite_positions = []
for lat, lon in satellites:
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    x = r * np.cos(lat_rad) * np.cos(lon_rad)
    y = r * np.cos(lat_rad) * np.sin(lon_rad)
    z = r * np.sin(lat_rad)

    satellite_positions.append((lat, lon, x, y, z))

# 创建 DataFrame 并保存为 CSV
df = pd.DataFrame(satellite_positions, columns=["Latitude", "Longitude", "X", "Y", "Z"])
csv_filename = "satellite_positions.csv"
df.to_csv(csv_filename, index=False)
print(f"卫星坐标已保存至 {csv_filename}")
