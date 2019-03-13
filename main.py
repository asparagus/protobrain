#!/usr/bin/python
# -*- coding: utf-8 -*-
import time
from encoders.numerical import CyclicEncoder
from brain.brain import Brain
from brain.computation import SparseComputation
from brain.learning import HebbianLearning
from brain.neocortex import SimpleNeocortexFactory
from brain.sensor import Sensor
from brain.synapses import InputConnection
from brain.synaptic_policy import ForwardSynapticPolicy
from brain.synaptic_policy import NullSynapticPolicy


if __name__ == '__main__':
    num_inputs = 100
    cycle = CyclicEncoder(1, 101, num_inputs, 0.04)
    sensor = Sensor(num_inputs)

    num_layers = 3
    num_neurons = 30
    neocortex_factory = SimpleNeocortexFactory(
        num_layers=num_layers,
        num_neurons=num_neurons,
        computation=SparseComputation(5),
        synaptic_policies=[ForwardSynapticPolicy(InputConnection)],
        learning_function=None
    )

    neocortex = neocortex_factory()
    brain = Brain(neocortex, sensor)

    # for _ in range(100):
    for i in range(1, 6):
        sensor.set_values(cycle.encode(i))
        neocortex.process()

        print(sensor)
        print(neocortex)
        time.sleep(0.5)
