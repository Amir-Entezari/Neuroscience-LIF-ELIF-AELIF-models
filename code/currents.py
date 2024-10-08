import math

import numpy
import torch
from pymonntorch import *


class ConstantCurrent(Behavior):
    def initialize(self, ng):
        self.value = self.parameter("value", None, required=True)
        self.noise_range = self.parameter("noise_range", 0.0)
        ng.I = ng.vector(self.value)

    def forward(self, ng):
        ng.I = ng.vector(self.value)
        self.add_noise(ng)

    def add_noise(self, ng):
        ng.I += (ng.vector("uniform") - 0.5) * self.noise_range


class StepCurrent(Behavior):
    def initialize(self, ng):
        self.value = self.parameter("value", None, required=True)
        self.t_start = self.parameter("t_start", required=True)
        self.t_end = self.parameter("t_end", None)
        self.noise_range = self.parameter("noise_range", 0.0)

        ng.I = ng.vector()

    def forward(self, ng):
        ng.I = ng.vector(0.0)
        if ng.network.iteration * ng.network.dt >= self.t_start:
            ng.I = ng.vector(mode=self.value)
            if self.t_end:
                if ng.network.iteration * ng.network.dt >= self.t_end:
                    ng.I = ng.vector(0.0)

        self.add_noise(ng)

    def add_noise(self, ng):
        ng.I += (ng.vector("uniform") - 0.5) * self.noise_range


class SinCurrent(Behavior):
    def initialize(self, ng):
        self.amplitude = self.parameter("amplitude", None, required=True)
        self.frequency = self.parameter("frequency", None, required=True)
        self.phase = self.parameter("phase", 0.0)
        self.offset = self.parameter("offset", 0.0)
        self.noise_range = self.parameter("noise_range", 0.0)

        t = ng.network.iteration * ng.network.dt
        ng.I = ng.vector()

    def forward(self, ng):
        t = ng.network.iteration * ng.network.dt
        ng.I = torch.sin(ng.vector(self.frequency * t) + self.phase) * self.amplitude + self.offset
        self.add_noise(ng)

    def add_noise(self, ng):
        ng.I += (ng.vector("uniform") - 0.5) * self.noise_range


class RampCurrent(Behavior):
    def initialize(self, ng):
        self.slope = self.parameter("slope", None, required=True)
        self.noise_range = self.parameter("noise_range", 0.0)

        ng.I = ng.vector()

    def forward(self, ng):
        t = ng.network.dt
        ng.I += self.slope * t
        self.add_noise(ng)

    def add_noise(self, ng):
        ng.I += (ng.vector("uniform") - 0.5) * self.noise_range


class ExpCurrent(Behavior):
    def initialize(self, ng):
        self.base = self.parameter("base")
        ng.I = ng.vector()

    def forward(self, ng):
        t = ng.network.iteration
        ng.I = self.base + torch.exp(ng.vector(t))

class LogCurrent(Behavior):
    def initialize(self, ng):
        self.horizontal_shift = self.parameter("horizontal_shift", 0.0)
        self.vertical_shift = self.parameter("vertical_shift", 0.0)
        self.noise_range = self.parameter("noise_range", 0.0)

        ng.I = ng.vector()

    def forward(self, ng):
        t = ng.network.iteration * ng.network.dt
        ng.I = self.vertical_shift + torch.log(ng.vector(t + self.horizontal_shift))
        self.add_noise(ng)

    def add_noise(self, ng):
        ng.I += (ng.vector("uniform") - 0.5) * self.noise_range


class NoisyCurrent(Behavior):
    def initialize(self, ng):
        self.iterations = self.parameter("iterations", None, required=True)
        self.noise_type = self.parameter("noise_type", "white")
        self.mean = self.parameter("mean", 0.0)
        self.std = self.parameter("std", 0.0)
        self.seed = self.parameter("seed", None)

        ng.I = ng.vector()

    def forward(self, ng):
        ng.I = ng.vector(float(self.add_noise(mean=self.mean,
                                              std=self.std,
                                              size=self.iterations)[ng.network.iteration]))

    def add_noise(self, mean, std, size):
        if self.seed is not None:
            self.set_seed()
        if self.noise_type == 'white':
            return self.white_noise(mean, std, size)
        elif self.noise_type == 'brownian':
            return self.brownian_noise(mean, std, size)
        else:
            raise ValueError("Unsupported noise type")

    def set_seed(self):
        np.random.seed(self.seed)

    def white_noise(self, mean, std, size):
        self.set_seed()
        return np.random.normal(mean, std, size)

    def brownian_noise(self, mean, std, size):
        # Generate white noise with mean=0 and std=1
        white_noise = np.random.normal(0, 1, size)

        # Generate cumulative sum to simulate Brownian motion
        brownian_motion = np.cumsum(white_noise)

        # Adjust mean and std
        adjusted_brownian_motion = (brownian_motion - np.mean(brownian_motion)) / np.std(brownian_motion)

        # Scale to desired mean and std
        scaled_brownian_noise = adjusted_brownian_motion * std + mean

        return scaled_brownian_noise


class RefractoryPeriod(Behavior):
    def initialize(self, ng):
        self.refractory_T = self.parameter("refractory_T", None, required=True) / ng.network.dt

        if not hasattr(ng, 'last_spike'):
            ng.last_spike = ng.vector(-self.refractory_T - 1)

    def forward(self, ng):
        ng.I = ng.I * (ng.last_spike < ng.network.iteration - self.refractory_T).byte()
