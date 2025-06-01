import numpy as np
from Timer import Timer


class Synapse:
    w_max = 1
    w_min = 0
    timer = Timer()

    def __init__(self, neuron, weight = 1.):
        self.neuron = neuron
        neuron.synapses.append(self)
        self.weight = weight
        self.synaptic_resource = np.array([])

    def spike(self):
        self.neuron.u += max(0., self.weight) / (self.w_max + max(0., self.weight))
        self.synaptic_resource = np.append(self.synaptic_resource, self.timer.time)

    def punish(self):
        self.weight -= len(self.synaptic_resource[self.timer.time - self.synaptic_resource < 0.05])*0.003

    def promotion(self):
        self.weight += len(self.synaptic_resource[self.timer.time - self.synaptic_resource < 0.05]) * 0.1


def create_synapses_with_layers(popul_1, popul_2, mode = "full"):
    if mode == "full":
        for neuron_1 in popul_1:
            for neuron_2 in popul_2:
                neuron_1.add_synaps(neuron_2)
