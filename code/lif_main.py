from pymonntorch import *
import torch


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
