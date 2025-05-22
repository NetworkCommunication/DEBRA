import math

def calculate_delay(distance=500e3, frequency=200e12, tx_gain=30, rx_gain=30, data_size=0.41 , bandwidth=2*10**7 , noise_density=1e-20, tx_power=20):
    """
    计算通信时延，包括传播时延和传输时延
    参数：
    - distance: 卫星间距离 (米)
    - frequency: 信号频率 (Hz)
    - tx_gain: 发射端增益 (dBi)
    - rx_gain: 接收端增益 (dBi)
    - data_size: 数据量 (GB)
    - bandwidth: 信道带宽 (Hz)
    - noise_density: 噪声功率谱密度 (W/Hz)
    - tx_power: 发射端功率 (W)

    返回：
    - 总通信时延 (秒)
    """

    c = 3e8

    # 计算波长 λ = c / f
    wavelength = c / frequency

    tx_gain_linear = 10 ** (tx_gain / 10)
    rx_gain_linear = 10 ** (rx_gain / 10)

    pr = (tx_power * tx_gain_linear * rx_gain_linear * wavelength**2) / (4 * math.pi * distance)**2

    R = bandwidth * math.log2(1 + pr / (noise_density * bandwidth))

    T_prop = distance / c

    T_trans = data_size / R

    total_delay = T_prop + T_trans

    return total_delay

if __name__ == '__main__':
    distance = 500e3          # 500 km
    # 计算时延
    total_delay = calculate_delay(distance)

    print(f"总通信时延: {total_delay:.6f} s")
