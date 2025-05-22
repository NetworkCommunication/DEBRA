import copy
import csv
from tools import *
from env import *
from utils import *

# 目标函数：计算训练时延
def objective_function(position):
    # print(position)
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
    # 计算通信时延
    total_comm_time = get_trans_delay(pos)
    # print(total_comm_time)
    return total_time + total_comm_time


# 多目标粒子群优化算法
class Particle:
    def __init__(self, num_devices, num_segments):
        # 初始化粒子的分割点位置，表示每个子模型分配到哪个设备
        self.position = self.generate_increasing_position(num_devices, num_segments)
        self.velocity = self.increasing_velocity(num_segments)  # 初始化速度
        self.best_position = np.copy(self.position)  # 初始化局部最优解
        self.best_value = float('inf')  # 最优解对应的值

    def generate_increasing_position(self, num_devices, num_segments):
        """
        生成递增的设备分配。
        num_devices: 总设备数
        num_segments: 子模型数
        """
        while True:
            # 随机生成递增的设备编号
            position = np.zeros(num_segments, dtype=int)
            position[0] = np.random.randint(1, num_devices/2 + 1)  # 随机选择第一个设备编号

            for i in range(1, num_segments):
                # 保证设备号递增
                position[i] = np.random.randint(position[i-1], num_devices)
            if objective_function(position) < float('inf'):
                break
        return position

    def increasing_velocity(self, num_segments):
        # 初始化一个6维向量，元素为0到3之间的整数
        vector = [random.randint(0, 3) for _ in range(num_segments)]

        # 对向量进行排序以确保非递减顺序
        vector.sort()
        return vector

class MOPSO:
    def __init__(self, objective_function, num_particles, num_devices, num_segments, max_iter):
        self.objective_function = objective_function
        self.num_particles = num_particles
        self.num_devices = num_devices
        self.num_segments = num_segments
        self.max_iter = max_iter
        self.particles = [Particle(num_devices, num_segments) for _ in range(num_particles)]
        self.global_best_position = None
        self.global_best_value = float('inf')

    def update_parameters(self, current_iter):
        # 初始化参数
        w_start = 1.0
        w_end = 0.5
        c1_start = 2.5
        c1_end = 0.5
        c2_start = 0.5
        c2_end = 2.5
        max_iterations = self.max_iter
        # 计算当前参数值
        progress = current_iter / max_iterations
        w = np.full(self.num_segments, w_start - (w_start - w_end) * progress)
        c1 = np.full(self.num_segments, c1_start - (c1_start - c1_end) * progress)
        c2 = np.full(self.num_segments, c2_start + (c2_end - c2_start) * progress)
        return w, c1, c2

    def optimize(self):
        for t in range(self.max_iter):
            w, c1, c2 = self.update_parameters(t)
            for particle in self.particles:
                # 计算适应度
                current_value = self.objective_function(particle.position)

                # 更新局部最优解
                if current_value < particle.best_value:
                    particle.best_value = current_value
                    particle.best_position = np.copy(particle.position)

                # 更新全局最优解
                if current_value < self.global_best_value:
                    self.global_best_value = current_value
                    self.global_best_position = np.copy(particle.position)

                # 更新粒子速度和位置
                # 生成两个随机数 r1 和 r2
                r1 = np.random.rand(self.num_segments)
                r2 = np.random.rand(self.num_segments)

                # 更新速度
                particle.velocity = (w * particle.velocity
                                 + c1 * r1 * (particle.best_position - particle.position)
                                 + c2 * r2 * (self.global_best_position - particle.position))

                # 速度裁剪，避免速度过大
                particle.velocity = np.clip(particle.velocity, 1, self.num_devices)

                # 更新位置
                particle.position = (particle.position + particle.velocity)/2

                # 位置裁剪，确保位置在合理的范围内
                particle.position = np.clip(particle.position, 1, self.num_devices)

                particle.position = np.round(particle.position).astype(int)

                # 保证位置是递增的且最后一位小于等于num_devices
                for i in range(1, self.num_segments):
                    particle.position[i] = np.maximum(particle.position[i], particle.position[i - 1])
                    if particle.position[i] > self.num_devices:
                        particle.position[i] = self.num_devices

            if (t % 30 == 0) or (t == self.max_iter - 1):
                with open("data/results.csv", mode="a", newline="") as file:
                    writer = csv.writer(file)
                    # 打印当前迭代的最佳值
                    writer.writerow(["MOPSO", t, self.global_best_value])
                    # print(f"Iteration {i}, Best Value: {self.alpha_value}")

        return self.global_best_position, self.global_best_value

def main(dev, sm, bw, iterations):

    global devices, submodels, bandwidth
    devices = dev
    submodels = sm
    bandwidth = bw
    num_devices = len(devices)
    # 计算最优分割点数k
    # 初始化PSO优化器
    pso = MOPSO(objective_function, num_particles=30, num_devices=num_devices, num_segments=len(submodels), max_iter=iterations)
    # 开始优化
    return pso.optimize()


if __name__ == "__main__":
    devices = get_devices()  # 获取卫星算力及资源
    num_devices = len(devices)
    # 计算最优分割点数k

    submodels = get_submodels()  # 获取最小子模型参数
    # 获取星间链路带宽（用于计算通信时延）
    bandwidth = get_bandwidth()
    # 初始化PSO优化器
    pso = MOPSO(objective_function, num_particles=30, num_devices=num_devices, num_segments=6, max_iter=500)
    # 开始优化
    best_position, best_value = pso.optimize()
    print("最佳分割点位置：", best_position)
    print("最佳时延：", best_value)

