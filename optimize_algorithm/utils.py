
# 计算每个子模型的计算时延
def compute_time(all_flo, device):
    return all_flo / (device['compute_capacity'])  # FLOPs / TFLOPs -> seconds


# 判断设备是否超出资源限制
def check_constraints(all_memory, device):
    # 检查内存是否满足约束
    return all_memory < device['memory_capacity']

