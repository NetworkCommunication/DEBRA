import copy
import csv
from tools import *
import numpy as np
from env import *
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


# 人工蜂群算法
class ArtificialBeeColony:
    def __init__(self, objective_function, num_bees, num_devices, num_segments, max_iter):
        self.objective_function = objective_function
        self.num_bees = num_bees
        self.num_devices = num_devices
        self.num_segments = num_segments
        self.max_iter = max_iter
        self.food_sources = [self.generate_random_position() for _ in range(num_bees)]
        self.food_values = [self.objective_function(pos) for pos in self.food_sources]
        self.best_position = self.food_sources[np.argmin(self.food_values)]
        self.best_value = min(self.food_values)

    def generate_random_position(self):
        position = np.zeros(self.num_segments, dtype=int)
        position[0] = np.random.randint(1, self.num_devices /2 + 1)
        for i in range(1, self.num_segments):
            position[i] = np.random.randint(position[i - 1], self.num_devices)
        return position

    def optimize(self):
        for t in range(self.max_iter):
            # 雇佣蜂阶段
            for i in range(self.num_bees):
                candidate = self.generate_neighbor(self.food_sources[i])
                candidate_value = self.objective_function(candidate)
                if candidate_value < self.food_values[i]:
                    self.food_sources[i] = candidate
                    self.food_values[i] = candidate_value

            # 观察蜂阶段
            probabilities = self.calculate_probabilities()
            for i in range(self.num_bees):
                if random.random() < probabilities[i]:
                    candidate = self.generate_neighbor(self.food_sources[i])
                    candidate_value = self.objective_function(candidate)
                    if candidate_value < self.food_values[i]:
                        self.food_sources[i] = candidate
                        self.food_values[i] = candidate_value

            # 侦查蜂阶段
            for i in range(self.num_bees):
                if self.food_values[i] == float('inf'):  # 重新生成位置
                    self.food_sources[i] = self.generate_random_position()
                    self.food_values[i] = self.objective_function(self.food_sources[i])

            # 更新全局最优解
            current_best_value = min(self.food_values)
            if current_best_value < self.best_value:
                self.best_value = current_best_value
                self.best_position = self.food_sources[np.argmin(self.food_values)]

            if (t % 30 == 0) or (t == self.max_iter - 1):
                with open("data/results.csv", mode="a", newline="") as file:
                    writer = csv.writer(file)
                    # 打印当前迭代的最佳值
                    writer.writerow(["ABC", t, self.best_value])
                    # print(f"Iteration {t}, Best Value: {self.best_value}")

        return self.best_position, self.best_value

    def generate_neighbor(self, position):
        neighbor = np.copy(position)
        idx = np.random.randint(0, self.num_segments)
        neighbor[idx] = np.clip(neighbor[idx] + np.random.randint(-1, 2), 1, self.num_devices)
        for i in range(1, self.num_segments):
            neighbor[i] = max(neighbor[i], neighbor[i - 1])
        return neighbor

    def calculate_probabilities(self):
        fitness = np.array([1.0 / (1.0 + val) if val != float('inf') else 0 for val in self.food_values])
        return fitness / fitness.sum()


def main(dev, sm, bw, iterations):
    global devices, submodels, bandwidth
    devices = dev
    submodels = sm
    bandwidth = bw
    num_devices = len(devices)

    # 初始化ABC优化器
    abc = ArtificialBeeColony(objective_function, num_bees=10, num_devices=num_devices, num_segments=len(submodels), max_iter=iterations)
    # 开始优化
    return abc.optimize()


if __name__ == "__main__":
    devices = get_devices()  # 获取设备资源
    num_devices = len(devices)
    submodels = get_submodels()  # 获取子模型
    bandwidth = get_bandwidth()  # 获取带宽
    # 初始化ABC优化器
    abc = ArtificialBeeColony(objective_function, num_bees=30, num_devices=num_devices, num_segments=len(submodels), max_iter=500)
    # 开始优化
    best_position, best_value = abc.optimize()
    print("最佳分割点位置：", best_position)
    print("最佳时延：", best_value)
