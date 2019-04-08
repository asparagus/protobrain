#!/usr/bin/python
# -*- coding: utf-8 -*-
import time
from encoders import numerical
from brain import brain
from brain import computation
from brain import neocortex
from brain import sensor
from brain import synapses
from brain import synaptic_policy


if __name__ == '__main__':
    num_inputs = 100
    cycle = numerical.CyclicEncoder(1, 101, num_inputs, 0.04)
    sens = sensor.Sensor(num_inputs)

    num_layers = 3
    num_neurons = 30
    neocortex_factory = neocortex.SimpleNeocortexFactory(
        num_layers=num_layers,
        num_neurons=num_neurons,
        computation=computation.SparseComputation(5),
        synaptic_policies=[synaptic_policy.ForwardSynapticPolicy(synapses.InputConnection)],
        learning_function=None
    )

    neoctx = neocortex_factory()
    brn = brain.Brain(neoctx, sens)

    for i in range(1, 6):
        sens.set_values(cycle.encode(i))
        neoctx.process()

        print(sens)
        print(neoctx)
        time.sleep(0.5)
