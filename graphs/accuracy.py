#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pylab
import TimeData
import json
import numpy as np


device = "980"

if device == "980":
    cpu = json.load(open("../../Resultados/" + str(TimeData.epochs) + "/GTX 980/mnist_cpu.json", "r"))
    gpu = json.load(open("../../Resultados/" + str(TimeData.epochs) + "/GTX 980/mnist_gpu.json", "r"))
elif device == "k40":
    cpu = json.load(open("../../Resultados/" + str(TimeData.epochs) + "/K 40/mnist_cpu.json", "r"))
    gpu = json.load(open("../../Resultados/" + str(TimeData.epochs) + "/K 9840/mnist_gpu.json", "r"))

accuracies_cpu = []
accuracies_gpu = []

macrof1_cpu = []
macrof1_gpu = []

confusion_matrix_cpu = cpu['testing']['confusion_matrix']
confusion_matrix_gpu = gpu['testing']['confusion_matrix']

for epoch in cpu['trainning']['epochs']:
    epoch = epoch['epoch']
    accuracies_cpu.append(float(epoch['accuracy']) * 100)
    confusion_matrix = np.asmatrix(epoch['confusion_matrix'])

    precisions = []
    recalls = []

    for i in range(len(confusion_matrix)):
        precisions.append(confusion_matrix[i, i] / (confusion_matrix[:, i].sum()))
        recalls.append(confusion_matrix[i, i] / (confusion_matrix[i, :].sum()))

    macroAvgPrecision = np.mean(precisions)
    macroAvgRecall = np.mean(recalls)

    macrof1_cpu.append(2 * (macroAvgPrecision * macroAvgRecall) / (macroAvgPrecision + macroAvgRecall))


for epoch in gpu['trainning']['epochs']:
    epoch = epoch['epoch']
    accuracies_gpu.append(float(epoch["accuracy"]) * 100)
    confusion_matrix = np.asmatrix(epoch['confusion_matrix'])

    precisions = []
    recalls = []

    for i in range(len(confusion_matrix)):
        precisions.append(confusion_matrix[i, i] / (confusion_matrix[:, i].sum()))
        recalls.append(confusion_matrix[i, i] / (confusion_matrix[i, :].sum()))

    macroAvgPrecision = np.mean(precisions)
    macroAvgRecall = np.mean(recalls)

    macrof1_gpu.append(2 * (macroAvgPrecision * macroAvgRecall) / (macroAvgPrecision + macroAvgRecall))

pylab.plot(range(0, TimeData.epochs), accuracies_cpu, '-b', label='CPU')
pylab.plot(range(0, TimeData.epochs), accuracies_gpu, '-r', label='GPU')
pylab.ylabel("Acurácia")
pylab.xlabel("Épocas")
pylab.title("Acurácia x Épocas")
pylab.legend(loc='lower right')
pylab.show()

pylab.plot(range(0, TimeData.epochs), macrof1_cpu, '-b', label='CPU')
pylab.plot(range(0, TimeData.epochs), macrof1_gpu, '-r', label='GPU')
pylab.ylabel("F1")
pylab.xlabel("Épocas")
pylab.title("F1 x Épocas")
pylab.legend(loc='lower right')
pylab.show()


pylab.imshow(confusion_matrix_cpu, cmap='hot')
pylab.show()

pylab.imshow(confusion_matrix_gpu, cmap='hot')
pylab.show()
