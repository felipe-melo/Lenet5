#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pylab
import TimeData


device = "k40"

if device == "980":
    cpu = open("../../Resultados/"+str(TimeData.epochs)+"/GTX 980/mnist_cpu.out", "r")
    gpu = open("../../Resultados/"+str(TimeData.epochs)+"/GTX 980/mnist_gpu.out", "r")
elif device == "k40":
    cpu = open("../../Resultados/"+str(TimeData.epochs)+"/K 40/mnist_cpu.out", "r")
    gpu = open("../../Resultados/"+str(TimeData.epochs)+"/K 40/mnist_gpu.out", "r")

accuracies_cpu = []
accuracies_gpu = []

for i in range(0, TimeData.epochs+1):
    line_cpu = cpu.readline()
    line_gpu = gpu.readline()
    if i == 0:
        continue

    accuracies_cpu.append(float(line_cpu.split("accuracy: ")[1]) * 100)
    accuracies_gpu.append(float(line_gpu.split("accuracy: ")[1]) * 100)

pylab.plot(range(0, TimeData.epochs), accuracies_cpu, '-b', label='CPU')
pylab.plot(range(0, TimeData.epochs), accuracies_gpu, '-r', label='GPU')
pylab.ylabel("Acurácia")
pylab.xlabel("Épocas")
pylab.title("Acurácia x Épocas")
pylab.legend(loc='lower right')
pylab.show()
