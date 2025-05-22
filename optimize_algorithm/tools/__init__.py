from .distance_cal import calculate_path_distance
from .delay_cal import calculate_delay

def get_trans_delay(pos):
    total_dist, expanded_path = calculate_path_distance(pos)
    total_delay = 0

    if total_dist != 0:
        total_delay = calculate_delay(int(total_dist*1000))

    return total_delay+ len(pos)*0.1