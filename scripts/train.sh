#!/usr/bin/env bash

set -e

# python train_nn.py -m fcn -l ctrossentropy -d sel-10
# python train_nn.py -m resnet -l ctrossentropy -d sel-10
python train_nn.py -m fcn -l focal -d sel-10
python train_nn.py -m resnet -l focal -d sel-10

# python train_nn.py -m fcn -l ctrossentropy -d all
# python train_nn.py -m resnet -l ctrossentropy -d all
# python train_nn.py -m fcn -l focal -d all
# python train_nn.py -m resnet -l focal -d all

shutdown.exe -s -t 0
