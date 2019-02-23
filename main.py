#!/usr/bin/python
# -*- coding: utf-8 -*-
from brain.minicolumn import SimpleMiniColumnFactory
from brain.layer import SimpleLayerFactory
from brain.neuron import SimpleNeuronFactory


if __name__ == '__main__':
    neuron = SimpleNeuronFactory(5)()
    minicolumn = SimpleMiniColumnFactory(4, 5)()
    layer = SimpleLayerFactory(10, 4, 5)()

    print(neuron)
    print(minicolumn)
    print(layer)
