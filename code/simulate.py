from pymonntorch import *
from plots import plot_membrane_potential
import torch
import matplotlib.pyplot as plt


class Simulation:
    def __init__(self, net: Network = None):
        self.net: Network
        if net:
            self.net = net
        else:
            self.net = Network()

    def add_neuron_group(self, tag, **kwargs):
        if tag in [ng.tag for ng in self.net.NeuronGroups]:
            raise Exception("The neuron group's id already exist.")
        NeuronGroup(net=self.net, tag=tag, **kwargs)

    def simulate(self, iterations=100):
        self.net.initialize()
        self.net.simulate_iterations(iterations=iterations)

    def plot_membrane_potential(self, title: str, model_idx: int = 3):
        num_ng = len(self.net.NeuronGroups)
        legend_position = (0, -0.2) if num_ng < 2 else (1.05, 1)
        # Generate colors for each neuron
        colors = plt.cm.jet(np.linspace(0, 1, num_ng))
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        for i, ng in enumerate(self.net.NeuronGroups):
            ax1.plot(ng.behavior[4].variables["u"][:, :1], color=colors[i], label='potential')
            ax2.plot(ng.behavior[4].variables["I"][:, :1], color=colors[i], label=f"{ng.tag} current")

            ax1.axhline(y=ng.behavior[model_idx].init_kwargs['threshold'], color='red', linestyle='--',
                        label=f'{ng.tag} Threshold')
            ax1.axhline(y=ng.behavior[model_idx].init_kwargs['u_reset'], color='black', linestyle='--',
                        label=f'{ng.tag} u_reset')

        ax1.set_xlabel('Time')
        ax1.set_ylabel('u(t)')
        ax1.set_title(f'Membrane Potential')
        ax1.legend(loc='upper left', bbox_to_anchor=legend_position, fontsize='small')

        ax2.set_xlabel('Time')
        ax2.set_ylabel("I(t)")
        ax2.set_title('Current')
        ax2.legend(loc='upper left', bbox_to_anchor=legend_position, fontsize='small')
        fig.suptitle(title)
        plt.tight_layout()

        plt.show()

    def plot_IF_curve(self, title: str=None, label:str=None, event_idx=5, current_idx=2, show=True):
        frequencies = []
        currents = []
        for i, ng in enumerate(self.net.NeuronGroups):
            spike_events = ng.behavior[event_idx].variables['spike']
            frequencies.append(len(spike_events))
            currents.append(ng.behavior[current_idx].init_kwargs['value'])
        plt.plot(currents, frequencies, label=label)
        plt.title(title)
        plt.xlabel('Current (I)')
        plt.ylabel('Frequency (f)')
        plt.legend()
        plt.grid(True)
        if show:
            plt.show()
        else:
            return plt


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
