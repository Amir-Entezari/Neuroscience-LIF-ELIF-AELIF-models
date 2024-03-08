import math

import torch
from pymonntorch import *


class ConstantCurrent(Behavior):
    def initialize(self, ng):
        self.value = self.parameter("value", None, required=True)
        ng.I = ng.vector(self.value)


class StepFunctionCurrent(Behavior):
    def initialize(self, ng):
        self.value = self.parameter("value")
        self.t0 = self.parameter("t0")
        ng.I = ng.vector()

    def forward(self, ng):
        if ng.network.iteration * ng.network.dt >= self.t0:
            ng.I = ng.vector(mode=self.value)


class SinCurrent(Behavior):
    def initialize(self, ng):
        self.amplitude = self.parameter("amplitude", None, required=True)
        self.frequency = self.parameter("frequency", None, required=True)
        self.phase = self.parameter("phase", 0.0)
        self.offset = self.parameter("offset", 0.0)

        t = ng.network.iteration * ng.network.dt
        ng.I = ng.vector()

    def forward(self, ng):
        t = ng.network.iteration * ng.network.dt
        ng.I = torch.sin(ng.vector(self.frequency * t) + self.phase) * self.amplitude + self.offset


class RampCurrent(Behavior):
    def initialize(self, ng):
        self.slope = self.parameter("slope", None, required=True)
        ng.I = ng.vector()

    def forward(self, ng):
        t = ng.network.dt
        ng.I += self.slope * t


# class ExpDecayCurrent(Behavior):
#     def initialize(self, ng):
#         self.base = self.parameter("base")
#         ng.I = ng.vector()
#
#     def forward(self, ng):
#         t = ng.network.iteration * ng.network.dt
#         ng.I = self.base + torch.exp(ng.vector(t))

class LogCurrent(Behavior):
    def initialize(self, ng):
        self.horizontal_shift = self.parameter("horizontal_shift", 0.0)
        self.vertical_shift = self.parameter("vertical_shift", 0.0)
        ng.I = ng.vector()

    def forward(self, ng):
        t = ng.network.iteration * ng.network.dt
        ng.I = self.vertical_shift + torch.log(ng.vector(t + self.horizontal_shift ))