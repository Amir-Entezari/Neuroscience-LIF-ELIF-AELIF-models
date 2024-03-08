from pymonntorch import *
from plots import plot_membrane_potential
import torch


class Simulation:
    def __init__(self, net: Network = None):
        self.net: Network
        if net:
            self.net = net
        else:
            self.net = Network()
        self.neuron_groups = {}

    def add_neuron_group(self, ng_id, **kwargs):
        if ng_id in self.neuron_groups.keys():
            raise Exception("The neuron group's id already exist.")
        self.neuron_groups[ng_id] = NeuronGroup(net=self.net, **kwargs)

    def simulate(self, iterations=100):
        self.net.initialize()
        self.net.simulate_iterations(iterations=iterations)


class SimulateNeuronGroup:
    def __init__(self, net: Network = None, **kwargs):
        self.net: Network
        if net:
            self.net = net
        else:
            self.net = Network()
        self.ng = NeuronGroup(
            net=self.net,
            **kwargs
        )

    def simulate(self, iterations=100):
        self.net.initialize()
        self.net.simulate_iterations(iterations=iterations)
