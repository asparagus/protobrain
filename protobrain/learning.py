#!/usr/bin/python
# -*- coding: utf-8 -*-
import abc
import numpy as np


class Learning(abc.ABC):

    @abc.abstractmethod
    def __call__(self, connections):
        raise NotImplementedError()


class HebbianLearning(Learning):
    def __init__(self, increase=0.05, decrease=0.002):
        self.increase = increase
        self.decrease = decrease

    def __call__(self, neurons):
        if neurons.passthrough:
            for layer in neurons.layers:
                self(layer)
            return

        output_values = neurons.output.values
        active_neurons = output_values == 1
        for input_name, input_unit in neurons.inputs.items():
            input_unit.synapses -= self.decrease
            input_unit.synapses[active_neurons,...] += self.increase
            input_unit.synapses = np.clip(input_unit.synapses, 0, 1)
