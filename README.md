# Reconhecimento de Dígitos Utilizando Redes Neurais em GPU

O [trabalho](https://github.com/felipe-melo/Lenet5/blob/master/TCC_Felipe_Melo.pdf) apresenta uma implementação de rede neural artificial para detecção de dígitos. A rede
foi baseada na [lennet5](http://yann.lecun.com/exdb/lenet/) apresentada em 1998 por [LeCun](http://yann.lecun.com/index.html). Com o avança das placas gráficas e da capacidade dos
computadores modernos é possível se acelerar o aprendizados de redes neurais, sem perda em termos de acurácia.

## Ferramentas utilizadas

O trabalho foi implementado em python com uso das bibliotecas [numpy](http://www.numpy.org/) e [theano](http://deeplearning.net/software/theano/), esta segunda permite a configuração
e envio de instruções para serem executadas em GPUs de maneira simples e bem transparente. Para execução em GPU é necessários que o drive da placa
de vídeo esteja instalado bem como a biblioteca C [CUDA](https://developer.nvidia.com/cuda-zone).

##Uso

O dataset utilziado para este trabalho foi o [MNIST](http://yann.lecun.com/exdb/mnist/), famosa base com imagens de dígitos de 0 a 9, dentro do package MNIST
no arquivo MNISTDataset.py[link], tem uma função, ```create_binary_files```, que cria um binário da base dados original, que deve estar
presenta no caminho informado no arquivo [Constants](https://github.com/felipe-melo/Lenet5/blob/master/util/Constants.py).

Parte da configuração da execução do theano deve estar no arquivo /home/user/.theanorc como abaixo, dependendo da versão da placa de vídeo e da biblioteca theano

```
[global]
floatX = float32
device = cuda0

[gpuarray]
preallocate = 1

[lib]
cnmem = 0.75
```

Para execução foram criados arquivos .sh com os comandos e parâmetros para o código. Nos comandos de execução algumas diretivas de CUDA para captura de informações da
execução podem ser usadas.

# Créditos de citação:
Se você utilizar este código em algum trabalho, por favor cite:

```Felipe Melo, Fellipe Duarte, Marcelo Zamith, Reconhecimento de Dígitos Utilizando Redes Neurais em GPU, 2017 ```
