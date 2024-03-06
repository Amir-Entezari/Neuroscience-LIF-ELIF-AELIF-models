from pymonntorch import *
import torch

import matplotlib.pyplot as plt

from hw1.code.currents import ConstantCurrent
from hw1.code.models import ELIF
from hw1.code.time_res import TimeResolution

net = Network(behavior={1: TimeResolution(dt=0.5), })

pop1 = NeuronGroup(
    net=net,
    size=200,
    behavior={
        2: ConstantCurrent(value=13.8),
        3: ELIF(R=1.7,
                tau=10,
                threshold=-13,
                rh_threshold=-42,
                u_rest=-65,
                u_reset=-73,
                delta_T=0.1
                ),
        4: Recorder(variables=["u", "I"], tag="ng1_rec"),
        5: EventRecorder(variables=['spike'], tag="ng1_event")
    }
)

net.initialize()
net.simulate_iterations(iterations=1000)

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
