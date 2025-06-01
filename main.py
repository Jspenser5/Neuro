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
    for i in range(100):
        encoded_population.append(SpikeEncoderNeuron(None))
    layer1_population = [] * 10
    layer1_1population = [] * 10
    for k in range(10):
        layer1_population.append(LIFNeuron())
    for k in range(10):
        layer1_1population.append(LIFNeuron())
    create_synapses_with_layers(encoded_population, layer1_population)
    create_synapses_with_layers(encoded_population, layer1_1population)
    output_neuron1 = [LIFNeuron(u_th=2, spike_records=[])]
    output_neuron2 = [LIFNeuron(u_th=2, spike_records=[])]
    create_synapses_with_layers(layer1_population, output_neuron1)
    create_synapses_with_layers(layer1_1population, output_neuron2)
    timer = Timer()
    for num in range(len(filepaths)):
        timer.time = 0.
        samplerate, signal = wavfile.read(filepaths[num])
        cwtmatr, freqs = pywt.cwt(signal, widths, waveletname, sampling_period=1/samplerate)
        cwtmatr = np.abs(cwtmatr)
        for i in range(len(cwtmatr)):
            cwtmatr[i] /= widths[i]
        a = spike_encoder(cwtmatr, u_th=100, tau=10)
        print(len(a))
        for i in range(len(a)):
            encoded_population[i].spike_signal = a[i]
        output_neuron1[0].clear()
        output_neuron2[0].clear()
        for neuron in layer1_population:
            neuron.clear()
        for neuron in layer1_1population:
            neuron.clear()
        for t in range(len(a[0])):
            for neuron in encoded_population:
                neuron.spike()
            for neuron in layer1_population:
                neuron.step()
                neuron.spike()
            for neuron in layer1_1population:
                neuron.step()
                neuron.spike()
            for neuron in output_neuron1:
                neuron.step()
            for neuron in output_neuron2:
                neuron.step()
            timer.time += 1/samplerate
            if output_neuron1[0].spike():
                if labels[num] == '1':
                    for neuron in layer1_population:
                        neuron.promotion_signal()
                    for neuron in output_neuron1:
                        neuron.promotion_signal()
                else:
                    for neuron in layer1_population:
                        neuron.punishment_signal()
                    for neuron in output_neuron1:
                        neuron.punishment_signal()
            if output_neuron2[0].spike():
                if labels[num] == '2':
                    for neuron in layer1_1population:
                        neuron.promotion_signal()
                    for neuron in output_neuron2:
                        neuron.promotion_signal()
                else:
                    for neuron in layer1_1population:
                        neuron.punishment_signal()
                    for neuron in output_neuron2:
                        neuron.punishment_signal()

    print("a")
    timer.time = 0.
    samplerate, signal = wavfile.read("resource/Дрон(0.0 0.5 и 0.7 0.8).wav")
    cwtmatr, freqs = pywt.cwt(signal, widths, waveletname, sampling_period=1 / samplerate)
    cwtmatr = np.abs(cwtmatr)
    for i in range(len(cwtmatr)):
        cwtmatr[i] /= widths[i]
    a = spike_encoder(cwtmatr, u_th=100, tau=10)
    for i in range(len(a)):
        encoded_population[i].spike_signal = a[i]
    output_neuron1[0].spike_records = []
    output_neuron2[0].spike_records = []
    for t in range(len(a[0])):
        for neuron in encoded_population:
            neuron.spike()
        for neuron in layer1_population:
            neuron.step()
            neuron.spike()
        for neuron in layer1_1population:
            neuron.step()
            neuron.spike()
        for neuron in output_neuron1:
            neuron.step()
        for neuron in output_neuron2:
            neuron.step()
        timer.time += 1 / samplerate

    plt.plot(np.linspace(0, len(a[0])/samplerate, num=len(a[0])),np.zeros(len(a[0])))
    span1 = plt.axvspan(0, 4.5, ymin=0, color='red', alpha = 0.2, label="Дрон тип 1")
    span2 = plt.axvspan(xmin=7, xmax=8.7, ymin=0, color='blue', alpha=0.2, label="Дрон тип 2")
    a = np.array(output_neuron1[0].spike_records)
    b = np.array(output_neuron2[0].spike_records)
    plt.vlines(a[a < 5.], ymin=0, ymax=1, color='r')
    plt.vlines(a[a > 6.], ymin=0, ymax=1, color='b')
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

