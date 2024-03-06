import torch
from pymonntorch import *


class LIF(Behavior):
    def initialize(self, ng):
        """
        Initialize the neurons
        :param ng: neuron group
        :return: None
        """
        # initial parameters in LIF model
        self.R = self.parameter("R", None, required=True)
        self.tau = self.parameter("tau", None, required=True)
        self.u_rest = self.parameter("u_rest", None, required=True)
        self.u_reset = self.parameter("u_reset", None, required=True)
        self.threshold = self.parameter("threshold", None, required=True)
        self.ratio = self.parameter("ratio", 1.1)
        # initial value of u in neurons
        ng.u = ng.vector("uniform") * (self.threshold - self.u_reset) * self.ratio
        ng.u += self.u_reset
        ng.spike = ng.u > self.threshold
        ng.u[ng.spike] = self.u_reset

    def forward(self, ng):
        """
        Apply LIF dynamic to neuron groups
        :param ng: neuron group
        :return: None
        """
        # Neuron dynamic
        leakage = ng.u - self.u_rest
        inp_u = self.R * ng.I
        ng.u += ((-leakage + inp_u) / self.tau) * ng.network.dt
        # Firing
        ng.spike = ng.u > self.threshold
        # Reset
        ng.u[ng.spike] = self.u_reset


class ELIF(Behavior):
    def initialize(self, ng):
        self.R = self.parameter("R", None, required=True)
        self.tau = self.parameter("tau", None, required=True)
        self.u_rest = self.parameter("u_rest", None, required=True)
        self.u_reset = self.parameter("u_reset", None, required=True)
        self.ratio = self.parameter("ratio", 1.0, required=True)
        self.threshold = self.parameter("threshold", None, required=True)
        self.rh_threshold = self.parameter("rh_threshold", None, required=True)
        self.delta_T = self.parameter("delta_T", None, required=True)

        ng.u = ng.vector("uniform") * (self.threshold - self.u_reset) * self.ratio
        ng.u += self.u_reset
        ng.spike = ng.u > self.threshold
        ng.u[ng.spike] = self.u_reset

    def forward(self, ng):
        # Neuron dynamic
        F = self.F(ng.u)
        inp_u = self.R * ng.I
        ng.u += ((F + inp_u) / self.tau) * ng.network.dt

        # Firing
        ng.spike = ng.u > self.threshold

        # Reset
        ng.u[ng.spike] = self.u_reset

    def F(self, u):
        leakage = u - self.u_rest
        return -leakage + self.delta_T * torch.exp((u - self.rh_threshold) / self.delta_T)
