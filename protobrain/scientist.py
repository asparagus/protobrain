#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Module for running experiments."""
from protobrain import protobrain
from protobrain import computation
from protobrain import neocortex
from protobrain import neuron
from protobrain import sensor
from protobrain import synapses
from protobrain import synaptic_policy
from protobrain.encoders import numerical
from protobrain.proto import snapshot_pb2
from protobrain.util import factory


class Scientist:

    def run(self, experiment):
        """Run an experiment."""
        num_inputs = experiment.sensor.representation_size
        cycle = numerical.CyclicEncoder(
            0,
            num_inputs,
            num_inputs,
            0.08
        )
        sens = sensor.Sensor(num_inputs)

        num_layers = len(experiment.cortex.layer)
        num_neurons = (n for n in experiment.cortex.layer)

        neuron_factory = neuron.NeuronsFactory(
            number_factory=factory.SequenceFactory(num_neurons),
            computation_factory=factory.ConstantFactory(
                computation.SparseComputation(5)
            )
        )
        neocortex_factory = neocortex.NeocortexFactory(
            num_layers_factory=factory.ConstantFactory(num_layers),
            neuron_factory=neuron_factory,
            synaptic_policies=[
                synaptic_policy.ForwardSynapticPolicy(
                    synapses.InputConnection
                )
            ],
            learning_function=None
        )

        cortex = neocortex_factory()
        brain = protobrain.ProtoBrain(cortex, sens)

        snapshots = [None] * experiment.iters
        curr = 0
        while curr < experiment.iters:
            value = curr % (num_inputs + 1)
            sens.set_values(cycle.encode(value))
            cortex.process()
            snapshot = brain.snapshot()

            snapshots[curr] = snapshot
            curr += 1

        history = snapshot_pb2.History()
        history.snapshot.extend(snapshots)
        return history
