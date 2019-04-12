#!/usr/bin/python
# -*- coding: utf-8 -*-
# import time
# from protobrain.proto import experiment_pb2
# from protobrain import scientist
from protobrain import brain
from protobrain import computation
from protobrain import neuron
from protobrain import sensor
from protobrain.encoders import numerical

if __name__ == '__main__':
    senz = sensor.Sensor(
        numerical.CyclicEncoder(
            min_value=0,
            max_value=100,
            length=98,
            sparsity=0.03
        ))

    layers = [neuron.Neurons(n) for n in [40, 40, 30, 20]]

    brain = brain.Brain(
        sensor=senz,
        neurons=neuron.FeedForward(layers)
    )

    for i in range(100):
        senz.feed(i)
        brain.compute(computation.StandardComputation(0.5))
