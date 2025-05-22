import pynvml
import time
import torch

def limit_gpu_tflops(target_tflops=2.0, gpu_index=0):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)


    base_tflops = 7.1
    base_GPU_frequency = 100
    target_ratio = target_tflops / base_tflops


    memory_clocks = pynvml.nvmlDeviceGetSupportedMemoryClocks(handle)
    print(memory_clocks)


    memory_clock = memory_clocks[4]

    supported_clocks = pynvml.nvmlDeviceGetSupportedGraphicsClocks(handle, memory_clock)
    print(supported_clocks)

    target_clock = int(base_GPU_frequency * target_ratio)  # 按比例选择频率
    target_clock = 300
    target_mem_clock = 7001
    # 锁定频率
    # pynvml.nvmlDeviceSetMemoryLockedClocks(handle, target_mem_clock, target_mem_clock)
    pynvml.nvmlDeviceSetGpuLockedClocks(handle, target_clock, target_clock)
    pynvml.nvmlDeviceSetPowerManagementLimit(handle, 125_000)  # 150W

    min_power = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)[0]  # 最低功率限制
    max_power = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)[1]  # 最高功率限制
    print(max_power)
    # 获取核心频率（单位：MHz）
    core_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
    # 获取显存频率（单位：MHz）
    memory_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)

    print(f"核心频率: {core_clock} MHz")
    print(f"显存频率: {memory_clock} MHz")

if __name__ == '__main__':
    # 调用函数限制算力
    limit_gpu_tflops(1000)

    # 恢复默认频率
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    pynvml.nvmlDeviceResetGpuLockedClocks(handle)

    pynvml.nvmlShutdown()
