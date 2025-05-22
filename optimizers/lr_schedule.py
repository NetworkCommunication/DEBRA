
def half_lr(init_lr, ep):
    lr = init_lr / 2**ep

    return lr

def step_lr(ep):
    if ep < 20:
        lr = 0.01
    elif ep < 40:
        lr = 0.001
    elif ep < 70:
        lr = 0.0005
    else:
        lr = 0.0001
    return lr
