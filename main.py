import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import  pywt
from SpikeEncoder import  spike_encoder, SpikeEncoderNeuron
from Synapses import create_synapses_with_layers
from matplotlib import colors


from Neurons import LIFNeuron
from Timer import Timer
from utils import read_file_datas

colors_list = ['#FFFFFF', '#000000']
cmap = colors.ListedColormap(colors_list)

def main(name):
    filepaths, labels = read_file_datas("resource/datas.txt")
    waveletname = 'cmor7.5-3.5'
    widths = np.linspace(50, 256, num=100)
    b = np.array([np.array([])] * 10)
    encoded_population = []
    layer1_population = [] * 10
    for k in range(10):
        layer1_population.append(LIFNeuron())
    create_synapses_with_layers(encoded_population, layer1_population)
    output_neuron = [LIFNeuron(u_th=2, spike_records=[]), LIFNeuron(u_th=2, spike_records=[])]
    create_synapses_with_layers(layer1_population, output_neuron)
    timer = Timer()

    for num in range(len(filepaths)):
        timer.time = 0.
        samplerate, signal = wavfile.read(filepaths[num])
        cwtmatr, freqs = pywt.cwt(signal, widths, waveletname, sampling_period=1/samplerate)
        print(len(cwtmatr))
        cwtmatr = np.abs(cwtmatr)
        print((len(cwtmatr)))
        for i in range(len(cwtmatr)):
            cwtmatr[i] /= widths[i]

        a = spike_encoder(cwtmatr, u_th=100, tau=10)
        for i in range(len(a)):
            encoded_population.append(SpikeEncoderNeuron(a[i]))
        for t in range(len(a[0])):
            for neuron in encoded_population:
                neuron.spike()
            for neuron in layer1_population:
                neuron.step()
                neuron.spike()
            for neuron in output_neuron:
                neuron.step()
            timer.time += 1/samplerate
            if output_neuron[0].spike() and labels[num] == 1:
                #pass
                for neuron in layer1_population:
                    neuron.promotion_signal()
                for neuron in output_neuron:
                    neuron.promotion_signal()
            elif output_neuron[1].spike() and labels[num] == 2:
                #pass
                for neuron in layer1_population:
                    neuron.promotion_signal()
                for neuron in output_neuron:
                    neuron.promotion_signal()
    print("a")
    timer.time = 0.
    for i in range(len(a)):
        encoded_population[i].spike_signal = a[i]
    output_neuron[0].spike_records = []
    for t in range(len(a[0])):
        for neuron in encoded_population:
            neuron.spike()
        for neuron in layer1_population:
            neuron.step()
            neuron.spike()
        for neuron in output_neuron:
            neuron.step()
        timer.time += 1 / samplerate

    plt.plot(np.linspace(0, len(a[0])/samplerate, num=len(a[0])),np.zeros(len(a[0])))
    span1 = plt.axvspan(0, 4.5, ymin=0, color='red', alpha = 0.2, label="Дрон тип 1")
    span2 = plt.axvspan(xmin=7, xmax=8.7, ymin=0, color='blue', alpha=0.2, label="Дрон тип 2")
    a = np.array(output_neuron[0].spike_records)
    b = np.array(output_neuron[1].spike_records)
    plt.vlines(a, ymin=0, ymax=1, color='r')
    plt.vlines(b, ymin=0, ymax=1, color='b')
    plt.axhline(0., color='b')
    plt.xlabel("Время (с.)")
    plt.legend(handles=[span1, span2],title="Маркеры наличия звука дронов на записи",
               fontsize=15, title_fontsize=15)

    # plt.plot(np.linspace(0, len(a[0]) / samplerate, num=len(a[0])),output_neuron[0].u_records)

    # plt.pcolormesh(np.linspace(0, len(a[0])/samplerate, num=len(a[0])),freqs, cwtmatr)
    # plt.colorbar()
    # plt.xlabel("Время (с.)")
    # plt.ylabel("Частота (Hz)")
    plt.show()



if __name__ == '__main__':
    main('PyCharm')

