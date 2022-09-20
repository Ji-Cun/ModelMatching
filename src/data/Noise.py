import numpy as np


def enlargeReductionNoise(data, r=0.1):
    size = len(data)
    if r<= 0:
        data = data
    elif r<=1:
        rate = np.random.uniform(low=(1 - r), high=(1 + r), size=size)
        data = data * rate
    else:
        rate = np.random.uniform(low=0, high=(1 + r), size=size)
        data = data * rate
    return data


def gaussianNoise(data, sigma=0.1):
    size = len(data)
    if sigma<=0:
        data=data
    else:
        noise = np.random.normal(loc=0, scale=sigma, size=size)
        data = data + noise
    return data
