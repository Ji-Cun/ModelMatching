import numpy as np
import math
from src.data.ConversionRelationships import conversion
from src.data.Noise import gaussianNoise, enlargeReductionNoise


class NormalDistributionModel:
    def __init__(self, mu=0, sigma=1, size=100):
        self.mu = mu
        self.sigma = sigma
        self.size = size

    def getModelData(self):
        data = np.random.normal(loc=self.mu, scale=self.sigma, size=self.size)
        return data


class UniformDistributionModel:
    def __init__(self, low=0, high=1, size=100):
        self.low = low
        self.high = high
        self.size = size

    def getModelData(self):
        data = np.random.uniform(low=self.low, high=self.high, size=self.size)
        return data


class Component:
    def __init__(self, id=0, size=100):
        self.id = id
        if id == 0:
            self.model = NormalDistributionModel(mu=0, sigma=1, size=size)
        if id == 1:
            self.model = NormalDistributionModel(mu=0, sigma=math.sqrt(2), size=size)
        if id == 2:
            self.model = NormalDistributionModel(mu=1, sigma=1, size=size)
        if id == 3:
            self.model = UniformDistributionModel(low=0, high=1, size=size)
        if id == 4:
            self.model = UniformDistributionModel(low=0, high=1.2, size=size)
        if id == 5:
            self.model = UniformDistributionModel(low=-0.2, high=1, size=size)

    def getData(self):
        data = self.model.getModelData()
        return data

    def getDataWithGaussianNoise(self, sigma=0.1):
        data = self.model.getModelData()
        data = gaussianNoise(data, sigma=sigma)
        return data

    def getDataWithEnlargeReductionNoise(self, r=0.1):
        data = self.model.getModelData()
        data = enlargeReductionNoise(data, r=r)
        return data

    def getDataWithBothNoise(self, sigma=0.1, r=0.1):
        data = self.model.getModelData()
        data = gaussianNoise(data, sigma=sigma)
        data = enlargeReductionNoise(data, r=r)
        return data


