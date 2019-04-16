#!/usr/bin/python
# -*- coding: utf-8 -*-
# import time
# from protobrain import scientist
from protobrain.proto.python import encoder_pb2
from protobrain.proto.python import experiment_pb2
from protobrain import brain
from protobrain import computation
from protobrain import learning
from protobrain import neuron
from protobrain import sensor
from protobrain.encoders import numerical

if __name__ == '__main__':
    # This should be received from somewhere
    exp = experiment_pb2.Experiment()
    exp.encoder.type = encoder_pb2.Encoder.NUMERICAL_CYCLIC
    exp.encoder.shape.extend([98])

    ext = encoder_pb2.CyclicEncoder.cyclic_encoder
    exp.encoder.Extensions[ext].min_value = 0
    exp.encoder.Extensions[ext].max_value = 100

    exp.cortex.layer.extend([40, 40, 30, 20])

    for i in range(100):
        inp = exp.input.add()
        inp.int = i

    # Actual code
    _computation = computation.SparseComputation(5)
    _learning = learning.HebbianLearning()

    encoder = None
    if exp.encoder.type == encoder_pb2.Encoder.NUMERICAL_CYCLIC:
        ext = encoder_pb2.CyclicEncoder.cyclic_encoder
        encoder = numerical.CyclicEncoder(
            length=exp.encoder.shape[0],
            min_value=exp.encoder.Extensions[ext].min_value,
            max_value=exp.encoder.Extensions[ext].max_value
        )
    elif exp.encoder.type == encoder_pb2.Encoder.NUMERICAL_SIMPLE:
        ext = encoder_pb2.SimpleEncoder.simple_encoder
        encoder = numerical.SimpleEncoder(
            length=exp.encoder.shape[0],
            min_value=exp.encoder.Extensions[ext].min_value,
            max_value=exp.encoder.Extensions[ext].max_value
        )
    else:
        raise ValueError("Invalid configuration:\n" + str(exp.encoder))

    senz = sensor.Sensor(encoder)
    layers = [neuron.Neurons(n) for n in exp.cortex.layer]

    brain = brain.Brain(
        sensor=senz,
        neurons=neuron.FeedForward(layers)
    )

    for inp in exp.input:
        value = None
        if inp.HasField('int'):
            value = inp.int
        elif inp.HasField('float'):
            value = inp.float
        elif inp.HasField('text'):
            value = inp.text
        else:
            raise ValueError("Empty input")

        senz.feed(value)
        brain.compute(_computation)
        brain.learn(_learning)
