#!/usr/bin/python
# -*- coding: utf-8 -*-
import time
from encoders import numerical
from brain import brain
from brain import computation
from brain import neocortex
from brain import neuron
from brain import sensor
from brain import synapses
from brain import synaptic_policy
from brain import experiment_pb2
from util import factory


if __name__ == '__main__':
    # TODO(ariel): these should come from stdin
    exp = experiment_pb2.Experiment()
    exp.live = False
    exp.sensor.representation_size = 25
    exp.cortex.layer.extend([40, 40, 30, 20])

    #############################
    num_inputs = exp.sensor.representation_size
    cycle = numerical.CyclicEncoder(
        0,
        num_inputs,
        num_inputs,
        0.08
    )
    sens = sensor.Sensor(num_inputs)

    num_layers = len(exp.cortex.layer)
    num_neurons = (n for n in exp.cortex.layer)

    neuron_factory = neuron.NeuronsFactory(
        number_factory=factory.SequenceFactory(num_neurons),
        computation_factory=factory.ConstantFactory(
            computation.SparseComputation(5)
        )
    )
    neocortex_factory = neocortex.NeocortexFactory(
        num_layers_factory=factory.ConstantFactory(num_layers),
        neuron_factory=neuron_factory,
        synaptic_policies=[synaptic_policy.ForwardSynapticPolicy(synapses.InputConnection)],
        learning_function=None
    )

    cortex = neocortex_factory()
    brn = brain.Brain(cortex, sens)

    for i in range(10):
        for j in range(num_inputs + 1):
            sens.set_values(cycle.encode(i))
            cortex.process()
            snapshot = brn.snapshot()

            print(snapshot)
