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

    def plot_membrane_potential(self, title: str, model_idx: int = 3):
        net = self.net
        num_neurons = len(net.NeuronGroups)
        # Generate colors for each neuron
        colors = plt.cm.jet(np.linspace(0, 1, num_neurons))
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        for i, ng in enumerate(net.NeuronGroups):
            ax1.plot(ng.behavior[4].variables["u"][:, :1], color=colors[i], label='potential')
            ax1.axhline(y=ng.behavior[model_idx].init_kwargs['threshold'], linestyle='--',
                        label=f'{ng.tag} Threshold')
            ax1.axhline(y=ng.behavior[model_idx].init_kwargs['u_reset'], color='black', linestyle='--',
                        label=f'{ng.tag} u_reset')
            ax2.plot(ng.behavior[4].variables["I"][:, :1], color=colors[i], label=f"{ng.tag} current")

        ax1.set_xlabel('Time')
        ax1.set_ylabel('u(t)')
        ax1.set_title(f'Membrane Potential')
        ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='small')

        ax2.set_xlabel('Time')
        ax2.set_ylabel("I(t)")
        ax2.set_title('Current')
        ax2.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='small')
        fig.suptitle(title)
        plt.tight_layout()

        plt.show()


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
