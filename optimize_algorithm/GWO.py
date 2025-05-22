import copy
import csv
from tools import *
import numpy as np
import random
from env import get_env, get_R
from utils import *

# 灰狼优化算法

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

# 灰狼优化算法（GWO）类
class Wolf:
    def __init__(self, num_devices, num_segments):
        # 初始化狼的位置，表示每个子模型分配到哪个设备
        self.position = self.generate_increasing_position(num_devices, num_segments)

    def generate_increasing_position(self, num_devices, num_segments):
        """
        生成递增的设备分配。
        num_devices: 总设备数
        num_segments: 子模型数
        """
        while True:
            # 随机生成递增的设备编号
            position = np.zeros(num_segments, dtype=int)
            position[0] = np.random.randint(1, num_devices / 2 + 1)  # 随机选择第一个设备编号

            for i in range(1, num_segments):
                # 保证设备号递增
                position[i] = np.random.randint(position[i - 1], num_devices)
            if objective_function(position) < float('inf'):
                break
        return position


class GWO:
    def __init__(self, objective_function, num_wolves, num_devices, num_segments, max_iter):
        self.objective_function = objective_function
        self.num_wolves = num_wolves
        self.num_devices = num_devices
        self.num_segments = num_segments
        self.max_iter = max_iter
        self.wolves = [Wolf(num_devices, num_segments) for _ in range(num_wolves)]
        self.alpha_position = None
        self.alpha_value = float('inf')
        self.beta_position = None
        self.beta_value = float('inf')
        self.delta_position = None
        self.delta_value = float('inf')

    def optimize(self):
        for t in range(self.max_iter):
            # 更新Alpha、Beta、Delta狼
            for wolf in self.wolves:
                # 计算适应度
                current_value = self.objective_function(wolf.position)

                # 更新全局最优解（Alpha、Beta、Delta狼）
                if current_value < self.alpha_value:
                    self.delta_value = self.beta_value
                    self.delta_position = np.copy(self.beta_position)

                    self.beta_value = self.alpha_value
                    self.beta_position = np.copy(self.alpha_position)

                    self.alpha_value = current_value
                    self.alpha_position = np.copy(wolf.position)

                elif current_value < self.beta_value:
                    self.delta_value = self.beta_value
                    self.delta_position = np.copy(self.beta_position)

                    self.beta_value = current_value
                    self.beta_position = np.copy(wolf.position)

                elif current_value < self.delta_value:
                    self.delta_value = current_value
                    self.delta_position = np.copy(wolf.position)

            # 更新狼群的位置
            for wolf in self.wolves:
                a = 2 - t * (2 / self.max_iter)  # 逐渐减小
                r1 = np.random.rand(self.num_segments)
                r2 = np.random.rand(self.num_segments)

                A = 2 * a * r1 - a  # 用于平衡探索与开发(-a,a)
                C = 2 * r2  # 用于确定新的位置
                # print(self.alpha_position, self.beta_position, self.delta_position)

                # 更新狼的位置
                D_alpha = np.abs(C * self.alpha_position - wolf.position)
                D_beta = np.abs(C * self.beta_position - wolf.position)
                D_delta = np.abs(C * self.delta_position - wolf.position)

                # 根据头狼的位置和距离计算候选位置
                X1 = self.alpha_position - A * D_alpha
                X2 = self.beta_position - A * D_beta
                X3 = self.delta_position - A * D_delta

                w_start = 0.2
                w_end = 0.0
                max_iterations = self.max_iter
                # 计算当前参数值
                progress = t / max_iterations
                w = np.full(self.num_segments, w_start - (w_start - w_end) * progress)

                wolf.position = w * wolf.position + (X1 + X2 + X3) / 3.0

                # wolf.position = wolf.position + A * (D_alpha + D_beta + D_delta)

                # 保证位置合法，确保位置在合理范围内
                wolf.position = np.clip(wolf.position, 1, self.num_devices)

                wolf.position = np.round(wolf.position).astype(int) # 确保狼群编号是整数

                # 保证位置是递增的且最后一位小于等于num_devices
                for i in range(1, self.num_segments):
                    wolf.position[i] = np.maximum(wolf.position[i], wolf.position[i - 1])
                    if wolf.position[i] > self.num_devices:
                        wolf.position[i] = self.num_devices
            if (t % 30 == 0) or (t == self.max_iter - 1):
                print(f"Iteration {t}, Best Value: {self.alpha_value}")

        return self.alpha_position, self.alpha_value


def main(dev, sm, bw, iterations):
    global devices, submodels, bandwidth
    devices = dev
    submodels = sm
    bandwidth = bw
    num_devices = len(devices)
    num_segments = len(submodels)

    # 初始化GWO优化器
    gwo = GWO(objective_function, num_wolves=30, num_devices=num_devices, num_segments=num_segments, max_iter=iterations)
    # 开始优化
    return gwo.optimize()


if __name__ == "__main__":
    submodels, devices, bandwidth = get_env(sentinel=True)  # 获取卫星算力及资源

    # 运行GWO优化器
    best_position, best_value = main(devices, submodels, bandwidth, 100)
    print("最佳分割点位置：", best_position)
    print("最佳时延：", best_value)
