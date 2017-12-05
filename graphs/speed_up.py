#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import pylab
import json
import numpy as np


device = "k40"
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


labels = ['CPU', 'GPU']

loading_time_cpu = float(cpu['trainning']['loading_time'])
trainning_time_cpu = float(cpu['trainning']['time'])

loading_time_gpu = float(gpu['trainning']['loading_time'])
trainning_time_gpu = float(gpu['trainning']['time'])

N = 2
loading = (loading_time_cpu, loading_time_gpu)
trainning = (trainning_time_cpu, trainning_time_gpu)
ind = np.arange(N)
width = 0.4

p1 = pylab.bar(ind, trainning, width)
#p2 = pylab.bar(ind, loading, width, bottom=trainning, color='#d62728')

pylab.ylabel('Tempo')
pylab.title('Tempo de Execução')
pylab.xticks(ind, ('GPU', 'CPU'))

pylab.show()
