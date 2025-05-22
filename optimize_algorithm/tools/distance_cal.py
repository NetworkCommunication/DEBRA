import csv
import numpy as np
from scipy.spatial import distance
from collections import deque


# 读取设备坐标
def load_device_positions(csv_file):
    positions = {}  # 存储设备 {id: (x, y, z)}
    with open(csv_file, mode='r', newline='') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过表头
        for row_member, row in enumerate(reader, start=1):
            device_id = int(row_member)
            x, y, z = map(float, row[2:5])
            positions[device_id] = (x, y, z)
    return positions


# 设备邻接关系（假设某种拓扑结构，比如网格）
neighbors = {
    1: [2, 5, 6], 2: [3, 4, 5, 6], 3: [4, 5],
    4: [5, 8, 9], 5: [6, 7, 8, 9], 6: [7, 8],
    7: [8], 8: [9]
}

csv_file = "tools/satellite_positions.csv"  # 设备坐标文件
positions = load_device_positions(csv_file)

# 使用 BFS 查找两点之间的最短路径
def bfs_shortest_path(start, end):
    queue = deque([(start, [start])])  # (当前设备, 访问路径)

    while queue:
        current, path = queue.popleft()

        if current == end:
            return path  # 返回路径

        for neighbor in neighbors.get(current, []):
            queue.append((neighbor, path + [neighbor]))

    return None  # 无法到达


# 计算路径距离
def calculate_path_distance(path):
    total_distance = 0
    final_path = []  # 存储展开后的路径（每两点相邻）
    for i in range(len(path) - 1):
        start, end = path[i], path[i + 1]

        if end in neighbors[start]:  # 直接相邻
            final_path.append((start, end))
            total_distance += distance.euclidean(positions[start], positions[end])
        else:  # 不相邻，寻找最短路径
            shortest_path = bfs_shortest_path(start, end)
            if not shortest_path:
                raise ValueError(f"无法找到 {start} 到 {end} 的路径")

            for j in range(len(shortest_path) - 1):
                s, e = shortest_path[j], shortest_path[j + 1]
                final_path.append((s, e))
                total_distance += distance.euclidean(positions[s], positions[e])
    # if total_distance == 0:
    #     print(path)
    return total_distance, final_path

if __name__ == '__main__':
    path = [1,4 ]  # 输入设备序列

    total_dist, expanded_path = calculate_path_distance(path)

    # print(f"输入路径: {path}")
    # print(f"转换后路径: {expanded_path}")
    # print(f"总距离: {total_dist:.2f} km")
