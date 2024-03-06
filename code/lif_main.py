from pymonntorch import *
import torch

import matplotlib.pyplot as plt

from hw1.code.current import ConstantCurrent
from hw1.code.lif import LIF
from hw1.code.time_res import TimeResolution

net = Network(behavior={1: TimeResolution(dt=1.0), })

pop1 = NeuronGroup(
    net=net,
    size=200,
    behavior={
        2: ConstantCurrent(value=10),
        3: LIF(R=5,
               tau=10,
               threshold=-37,
               u_rest=-67,
               u_reset=-75,
               ),
        4: Recorder(variables=["u", "I"], tag="ng1_rec"),
        5: EventRecorder(variables=['spike'], tag="ng1_event")
    }
)

net.initialize()
net.simulate_iterations(iterations=100)

plt.plot(net["ng1_rec", 0].variables["u"][:, :3])
plt.show()

spike_events = net["ng1_event", 0].variables["spike"]

spike_times = spike_events[:, 0]
neuron_ids = spike_events[:, 1]
# Plot the raster plot
plt.figure(figsize=(8, 6))
plt.scatter(spike_times, neuron_ids, marker='|', color='blue', s=10)

plt.xlabel('Time')
plt.ylabel('Neuron ID')
plt.title('Raster Plot for LIF model')
plt.yticks(neuron_ids.unique())
plt.grid(True)
plt.show()
