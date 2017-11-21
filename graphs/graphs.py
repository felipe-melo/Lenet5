#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import re
import numpy as np
import TimeData


plt.rcdefaults()

device = "980"

#Variáveis
execution_data = {'DtoH': 0, 'Execução': 0, 'HtoD': 0, 'overhead': 0}

if device == "980":
    file = open("../../Resultados/30/GTX 980/mnist_cuda_profile.out", "r")
    cpu_time = TimeData.cpu_mnist_time_980
    gpu_time = TimeData.gpu_mnist_time_980
elif device == "k40":
    file = open("../../Resultados/30/K 40/MNIST_cuda_profiler", "r")
    cpu_time = TimeData.cpu_mnist_time_k40
    gpu_time = TimeData.gpu_mnist_time_k40

holding = 0
isFist = True
start = 0
end = 0
_sum = 0
#Variáveis

for i, lines in enumerate(file.readlines()):
    values = re.split(r'[ ][ ][ ]*', lines)

    if len(values) < 13 or values[1] == 'Start':
        continue

    if isFist:
        start = float(values[0])
        isFist = False

    _sum += float(values[1])
    end = float(values[0])

    if 'HtoD' in values[12]:
        execution_data['HtoD'] += float(values[1])
    elif 'DtoH' in values[12]:
        execution_data['DtoH'] += float(values[1])
    elif values[2] != '-':
        execution_data['Execução'] += float(values[1])

execution_data['overhead'] = end - start - _sum


def speedUp():
    N = 2
    cpus_time = (cpu_time, cpu_time)

    ind = np.arange(N)  # the x locations for the groups
    width = 0.35       # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, cpus_time, width, color='b')

    gpus_time = (gpu_time, (gpu_time - execution_data['overhead']))
    rects2 = ax.bar(ind + width, gpus_time, width, color='y')

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Tempo(s)')
    ax.set_title('MNIST')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(('GPU com Overhead', 'GPU sem Overhead'))

    ax.legend((rects1[0], rects2[0]), ('CPU', 'GPU'))


    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                    '%d' % int(height),
                    ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    plt.show()


def gpuUsage():
    y_pos = np.arange(len(execution_data))

    labels = list(execution_data.keys())
    sizes = list(execution_data.values())

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.show()

def main():
    speedUp()
    gpuUsage()


if __name__ == '__main__':
    main()
