import numpy as np

from Synapses import Synapse


def thrashhold_mapper(k, u_th):
    if k > u_th: return 1
    return 0

def spike_encoder(scalogramm, u_th = 100., tau = 5, delta_t = 0.01):
    spike_values = np.zeros(scalogramm.shape)
    for i in range(len(scalogramm)):
        u = 0.
        for j in range(len(scalogramm[i])):
            if u > u_th:
                spike_values[i, j] = 1.
                u = 0.
            u += (scalogramm[i, j] - u/tau)*delta_t
    return spike_values

class SpikeEncoderNeuron:

    w_max = 1.
    w_min = 0

    def __init__(self, spike_signal):
        self.spike_signal = spike_signal
        self.synaptic_connections = []
        self.synaptic_weights = []

    def add_synaps(self, neuron):
        self.synaptic_connections.append(Synapse(neuron, 0.5))

    def spike(self):
        if len(self.spike_signal) != 0 and self.spike_signal[0] == 1.:
            for connection in self.synaptic_connections:
                connection.spike()
        self.spike_signal = self.spike_signal[1:]
