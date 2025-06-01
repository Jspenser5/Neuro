import numpy as np

from Synapses import Synapse
from Timer import Timer

TAU = 0.5
TAU_TH = 0.5
DELTA_T = 0.01

class LIFNeuron:
    timer = Timer()
    def __init__(self, u = 0., u_th = 1., u_records = None, spike_records = None):
        self.u = u
        self.u_th = u_th
        self.u_records = u_records
        self.spike_records = spike_records
        self.synaptic_connections = []
        self.spiking = False
        self.synapses = []

    def step(self):
        if (self.u_records is not None):
            self.u_records = np.append(self.u_records, self.u)

        if(self.u >= self.u_th):
            self.u = 0.
            self.u_th += 0.5
            if(self.spike_records is not None):
                self.spike_records = np.append(self.spike_records, self.timer.time)
            self.spiking = True

        self.u -= (self.u/TAU)*DELTA_T

        if(self.u_th > 1.):
            self.u_th -= (self.u_th - 1.)/TAU_TH*DELTA_T
            if(self.u_th < 1.):
                self.u_th = 1.

    def add_synaps(self, neuron):
        self.synaptic_connections.append(Synapse(neuron))

    def spike(self):
        if self.spiking:
            self.spiking = False
            for connection in self.synaptic_connections:
                connection.spike()
            return True
        return False

    def punishment_signal(self):
        for synapse in self.synapses:
            synapse.punish()

    def promotion_signal(self):
        for synapse in self.synapses:
            synapse.promotion()

    def clear(self):
        self.spike_records = []
        for synapse in self.synapses:
            synapse.clear()
