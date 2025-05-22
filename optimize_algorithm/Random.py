import copy
import csv
from tools import *
import numpy as np
import random
from env import get_env, get_R
from utils import *

# 目标函数：计算训练时延
def objective_function(position):
    total_time = 0
    total_comm_time = 0
    num_submodels = len(position)
    R = get_R()  # 约束因子
    pos = []

    # 确定每个子模型的设备分配
    i=0
    while i < num_submodels:
        pos.append(position[i])
        all_flo = submodels[i]['flops']
        all_memory = submodels[i]['memory']
        device = devices[position[i]-1]  # 每个子模型分配的设备
        for j in range(num_submodels - i-1):
            if position[i] == position[i+1]:
                all_flo += submodels[i+1]['flops'] # 子模型在相同的设备，所需内存量相加
                all_memory += submodels[i+1]['memory']
                i += 1
            else:
                break
        i += 1
        # 计算计算时延
        if check_constraints(all_memory, device):
            total_time += compute_time(all_flo, device)
        else:
            total_time += compute_time(all_flo, device) + R * (all_memory - device['memory_capacity'])  # 如果超出约束，给出惩罚
    # print(pos)
    # 计算通信时延（假设每个子模型的大小等于内存大小）
    total_comm_time = get_trans_delay(pos)
    # print(total_comm_time)
    return total_time + total_comm_time


def generate_increasing_position(num_devices, num_segments, iterations):
    """
    生成递增的设备分配。
    num_devices: 总设备数
    num_segments: 子模型数
    """
    num = iterations
    best_position = []
    best_value = 99999
    for t in range(num):
        # 随机生成递增的设备编号
        position = np.zeros(num_segments, dtype=int)
        position[0] = np.random.randint(1, num_devices)  # 随机选择第一个设备编号

        for i in range(1, num_segments):
            # 保证设备号递增
            position[i] = np.random.randint(position[i-1], num_devices)

        # 保证位置合法，确保位置在合理范围内
        position = np.clip(position, 1, num_devices)
        position = np.round(position).astype(int)

        tmp = objective_function(position)
        if tmp < best_value:
            best_position = position
            best_value = tmp
        if (t % 30 == 0) or (t == num - 1):
            with open("data/results.csv", mode="a", newline="") as file:
                writer = csv.writer(file)
                # 打印当前迭代的最佳值
                writer.writerow(["RD", t, best_value])
                # print(f"Iteration {t}, Best Value: {self.alpha_value}")

    return best_position, best_value


def main(dev, sm, bw, iterations):
    global devices, submodels, bandwidth
    devices = dev
    submodels = sm
    bandwidth = bw

    return generate_increasing_position(len(devices), len(submodels), iterations)


if __name__ == "__main__":
    submodels, devices, bandwidth = get_env(sentinel=True)  # 获取卫星算力及资源

    best_position, best_value = main(devices, submodels, bandwidth)
    print("最佳分割点位置：", best_position)
    print("最佳时延：", best_value)
