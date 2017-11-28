#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import pylab
import json
import numpy as np


device = "710M"
dataset = "mnist"

if device == "980":
    cpu = json.load(open("../../Resultados/30/GTX 980/%s_cpu.json" % dataset, "r"))
    gpu = json.load(open("../../Resultados/30/GTX 980/%s_gpu.json" % dataset, "r"))
elif device == "k40":
    cpu = json.load(open("../../Resultados/30/K 40/%s_cpu.json" % dataset, "r"))
    gpu = json.load(open("../../Resultados/30/K 40/%s_gpu.json" % dataset, "r"))
elif device == "710M":
    cpu = json.load(open("../../Resultados/30/710M/%s_cpu.json" % dataset, "r"))
    gpu = json.load(open("../../Resultados/30/710M/%s_gpu.json" % dataset, "r"))

train_accuracies_cpu = []
train_accuracies_gpu = []

test_accuracies_cpu = []
test_accuracies_gpu = []

train_macrof1_cpu = []
test_macrof1_cpu = []

train_macrof1_gpu = []
test_macrof1_gpu = []

labels = ['Zero', 'Um', 'Dois', 'Três', 'Quatro', 'Cinco', 'Seis', 'Sete', 'Oito', 'Nove']

train_confusion_matrix = None
test_confusion_matrix = None

quant = len(cpu['trainning']['epochs'])

for epoch in cpu['trainning']['epochs']:
    epoch = epoch['epoch']
    train_accuracies_cpu.append(float(epoch['train_accuracy']) * 100)
    test_accuracies_cpu.append(float(epoch['test_accuracy']) * 100)

    train_confusion_matrix = np.asmatrix(epoch['train_confusion_matrix'])
    test_confusion_matrix = np.asmatrix(epoch['test_confusion_matrix'])
    
    precisions = []
    recalls = []

    for i in range(len(train_confusion_matrix)):
        precisions.append(train_confusion_matrix[i, i] / (train_confusion_matrix[:, i].sum()))
        recalls.append(train_confusion_matrix[i, i] / (train_confusion_matrix[i, :].sum()))

    macroAvgPrecision = np.mean(precisions)
    macroAvgRecall = np.mean(recalls)

    train_macrof1_cpu.append(2 * (macroAvgPrecision * macroAvgRecall) / (macroAvgPrecision + macroAvgRecall))

    precisions = []
    recalls = []

    for i in range(len(test_confusion_matrix)):
        precisions.append(test_confusion_matrix[i, i] / (test_confusion_matrix[:, i].sum()))
        recalls.append(test_confusion_matrix[i, i] / (test_confusion_matrix[i, :].sum()))

    macroAvgPrecision = np.mean(precisions)
    macroAvgRecall = np.mean(recalls)

    test_macrof1_cpu.append(2 * (macroAvgPrecision * macroAvgRecall) / (macroAvgPrecision + macroAvgRecall))


for epoch in gpu['trainning']['epochs']:
    epoch = epoch['epoch']
    train_accuracies_gpu.append(float(epoch["train_accuracy"]) * 100)
    test_accuracies_gpu.append(float(epoch["test_accuracy"]) * 100)

    train_confusion_matrix = np.asmatrix(epoch['train_confusion_matrix'])
    test_confusion_matrix = np.asmatrix(epoch['test_confusion_matrix'])

    precisions = []
    recalls = []

    for i in range(len(train_confusion_matrix)):
        precisions.append(train_confusion_matrix[i, i] / (train_confusion_matrix[:, i].sum()))
        recalls.append(train_confusion_matrix[i, i] / (train_confusion_matrix[i, :].sum()))

    macroAvgPrecision = np.mean(precisions)
    macroAvgRecall = np.mean(recalls)

    train_macrof1_gpu.append(2 * (macroAvgPrecision * macroAvgRecall) / (macroAvgPrecision + macroAvgRecall))

    precisions = []
    recalls = []

    for i in range(len(test_confusion_matrix)):
        precisions.append(test_confusion_matrix[i, i] / (test_confusion_matrix[:, i].sum()))
        recalls.append(test_confusion_matrix[i, i] / (test_confusion_matrix[i, :].sum()))

    macroAvgPrecision = np.mean(precisions)
    macroAvgRecall = np.mean(recalls)

    test_macrof1_gpu.append(2 * (macroAvgPrecision * macroAvgRecall) / (macroAvgPrecision + macroAvgRecall))

pylab.plot(range(quant), train_accuracies_gpu, '-r', label='Treino')
pylab.plot(range(quant), test_accuracies_gpu, '-b', label='Teste')
pylab.ylabel("Acurácia")
pylab.xlabel("Épocas")
pylab.title("Acurácia x Épocas (GPU)")
pylab.legend(loc='lower right')
pylab.show()

pylab.plot(range(quant), train_accuracies_cpu, '-r', label='Treino')
pylab.plot(range(quant), test_accuracies_cpu, '-b', label='Teste')
pylab.ylabel("Acurácia")
pylab.xlabel("Épocas")
pylab.title("Acurácia x Épocas (CPU)")
pylab.legend(loc='lower right')
pylab.show()

pylab.plot(range(quant), train_macrof1_gpu, '-b', label='Treino')
pylab.plot(range(quant), test_macrof1_gpu, '-r', label='Teste')
pylab.ylabel("F1")
pylab.xlabel("Épocas")
pylab.title("F1 x Épocas (GPU)")
pylab.legend(loc='lower right')
pylab.show()

pylab.plot(range(quant), train_macrof1_cpu, '-b', label='Treino')
pylab.plot(range(quant), test_macrof1_cpu, '-r', label='Teste')
pylab.ylabel("F1")
pylab.xlabel("Épocas")
pylab.title("F1 x Épocas (CPU)")
pylab.legend(loc='lower right')
pylab.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=pylab.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    pylab.imshow(cm, interpolation='nearest', cmap=cmap)
    pylab.title(title)
    pylab.colorbar()
    tick_marks = np.arange(len(classes))
    pylab.xticks(tick_marks, classes, rotation=45)
    pylab.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        pylab.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    pylab.tight_layout()
    pylab.ylabel('Previsão')
    pylab.xlabel('Valor real')

pylab.figure()
plot_confusion_matrix(train_confusion_matrix, classes=labels, title='Matriz de confusão (Treino)')

pylab.figure()
plot_confusion_matrix(test_confusion_matrix, classes=labels, title='Matriz de confusão (Teste)')

pylab.show()
