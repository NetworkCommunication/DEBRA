import random
import numpy as np

def get_devices():

    # 设备数量
    num_devices = 9
    devices = []
    for i in range(num_devices):
        compute_capacity = random.uniform(1.8, 2.2)
        memory_capacity = random.uniform(4, 8)
        devices.append({
            'id': i+1,
            'compute_capacity': compute_capacity,
            'memory_capacity': memory_capacity
        })

    # 输出设备信息
    for device in devices:
        print(f"设备 {device['id']} - 计算能力: {device['compute_capacity']} TFLOPs, 内存: {device['memory_capacity']} GB")

    return devices

def get_submodels():
    # 初始化k个子模型的FLOPs和内存需求
    submodels = []
    for i in range(16):
        # basicblock, 2 conv
        flops = round(random.uniform(2.3, 3.0), 2) # 每个子模型的TFLO ResNet18: 1.995
        memory = round(random.uniform(3.0, 3.6), 2) # 每个子模型的内存需求（单位GB）

        submodels.append({
            'flops': flops,
            'memory': memory
        })


    for i, submodel in enumerate(submodels):
        print(f"子模型 {i+1} - FLOPs: {submodel['flops']}, 内存: {submodel['memory']} GB")
    return submodels

def get_bandwidth():
    B_bandwidth = 10
    return B_bandwidth

def get_env(sentinel = False):
    if sentinel:
        return get_submodels(), get_devices(), get_bandwidth()
    raise ValueError("more times initial")

def get_R():
    # 惩罚因子
    R = 6
    return R