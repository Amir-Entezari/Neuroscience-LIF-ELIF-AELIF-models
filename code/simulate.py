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
        # NeuronGroup(net=self.net, tag=tag, **kwargs)
        SimulateNeuronGroup(net=self.net, tag=tag, **kwargs)

    def simulate(self, iterations=100):
        self.net.initialize()
        self.net.simulate_iterations(iterations=iterations)

    def plot_membrane_potential(self, title: str,
                                model_idx: int = 3,
                                record_idx=4,
                                save: bool = None,
                                filename: str = None):
        num_ng = len(self.net.NeuronGroups)
        legend_position = (0, -0.2) if num_ng < 2 else (1.05, 1)
        # Generate colors for each neuron
        colors = plt.cm.jet(np.linspace(0, 1, num_ng))
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        for i, ng in enumerate(self.net.NeuronGroups):
            ax1.plot(ng.behavior[record_idx].variables["u"][:, :1], color=colors[i], label=f'{ng.tag} potential')
            ax2.plot(ng.behavior[record_idx].variables["I"][:, :1], color=colors[i], label=f"{ng.tag} current")

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
        if save:
            plt.savefig(filename or title + '.pdf')
        plt.show()

    def plot_w(self, title: str,
               record_idx: int = 4,
               save: bool = None,
               filename: str = None):
        num_ng = len(self.net.NeuronGroups)
        legend_position = (0, -0.2) if num_ng < 2 else (1.05, 1)
        # Generate colors for each neuron
        colors = plt.cm.jet(np.linspace(0, 1, num_ng))
        for i, ng in enumerate(self.net.NeuronGroups):
            plt.plot(ng.behavior[record_idx].variables["w"][:, :1], color=colors[i], label=f'{ng.tag} adaptation')

        plt.xlabel('Time')
        plt.ylabel('w')
        plt.legend(loc='upper left', bbox_to_anchor=legend_position, fontsize='small')

        plt.title(title)
        if save:
            plt.savefig(filename or title + '.pdf')
        plt.show()

    def plot_IF_curve(self, title: str = None,
                      label: str = None,
                      event_idx=5,
                      current_idx=2,
                      show=True,
                      save: bool = None,
                      filename: str = None):
        frequencies = []
        currents = []
        for i, ng in enumerate(self.net.NeuronGroups):
            spike_events = ng.behavior[event_idx].variables['spike']
            frequencies.append(len(spike_events) / (self.net.network.dt * self.net.iteration))
            currents.append(ng.behavior[current_idx].init_kwargs['value'])
        plt.plot(currents, frequencies, label=label)
        plt.title(title)
        plt.xlabel('Current (I)')
        plt.ylabel('Frequency (f)')
        plt.legend()
        plt.grid(True)
        if save:
            plt.savefig(filename or title + '.pdf')
        if show:
            plt.show()
        else:
            return plt


class SimulateNeuronGroup(NeuronGroup):
    def plot_membrane_potential(self, title: str,
                                model_idx: int = 3,
                                record_idx=4,
                                save: bool = None,
                                filename: str = None):
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

        ax1.plot(self.behavior[record_idx].variables["u"][:, :1], label=f'potential')
        ax2.plot(self.behavior[record_idx].variables["I"][:, :1], label=f"current")

        ax1.axhline(y=self.behavior[model_idx].init_kwargs['threshold'], color='red', linestyle='--',
                    label=f'{self.tag} Threshold')
        ax1.axhline(y=self.behavior[model_idx].init_kwargs['u_reset'], color='black', linestyle='--',
                    label=f'{self.tag} u_reset')

        ax1.set_xlabel('Time')
        ax1.set_ylabel('u(t)')
        ax1.set_title(f'Membrane Potential')
        ax1.legend()

        ax2.set_xlabel('Time')
        ax2.set_ylabel("I(t)")
        ax2.set_title('Current')
        ax2.legend()
        fig.suptitle(title)
        plt.tight_layout()
        if save:
            plt.savefig(filename or title + '.pdf')
        plt.show()

    def plot_w(self, title: str,
               record_idx: int = 4,
               save: bool = None,
               filename: str = None):
        # Generate colors for each neuron
        plt.plot(self.behavior[record_idx].variables["w"][:, :1], label=f'adaptation')

        plt.xlabel('Time')
        plt.ylabel('w')
        plt.legend(loc='upper left', fontsize='small')

        plt.title(title)
        if save:
            plt.savefig(filename or title + '.pdf')
        plt.show()
