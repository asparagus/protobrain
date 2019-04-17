#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from protobrain import neuron
from protobrain.encoders import numerical
from protobrain.proto import encoder_pb2
from protobrain.proto import experiment_pb2
from protobrain.proto import sdr_pb2
from protobrain.proto import snapshot_pb2


def decode_encoder(encoder_proto):
    encoder = None
    if encoder_proto.type == encoder_pb2.Encoder.NUMERICAL_CYCLIC:
        ext = encoder_pb2.CyclicEncoder.cyclic_encoder
        encoder = numerical.CyclicEncoder(
            length=encoder_proto.shape[0],
            min_value=encoder_proto.Extensions[ext].min_value,
            max_value=encoder_proto.Extensions[ext].max_value
        )
    elif encoder_proto.type == encoder_pb2.Encoder.NUMERICAL_SIMPLE:
        ext = encoder_pb2.SimpleEncoder.simple_encoder
        encoder = numerical.SimpleEncoder(
            length=encoder_proto.shape[0],
            min_value=encoder_proto.Extensions[ext].min_value,
            max_value=encoder_proto.Extensions[ext].max_value
        )
    else:
        raise ValueError('Invalid configuration:\n' + str(encoder_proto))

    return encoder


def decode_neurons(neurons_proto):
    layers = [neuron.Neurons(n) for n in neurons_proto.layer]
    return neuron.FeedForward(layers)


def decode_input(input_proto):
    for inp in input_proto:
        yield getattr(inp, inp.WhichOneof('value'))


def encode_brain(brain, out=None):
    brain_snapshot = out or snapshot_pb2.Snapshot()
    encode_sensor(brain.sensor, brain_snapshot.sensor)
    encode_neurons(brain.neurons, brain_snapshot.cortex)
    return brain_snapshot


def encode_sensor(sensor, out=None):
    sensor_snapshot = out or snapshot_pb2.SensorSnapshot()
    encode_sdr(sensor.values, sensor_snapshot.sdr)

    return sensor_snapshot


def encode_neurons(neurons, out=None):
    cortex = out or snapshot_pb2.CortexSnapshot()
    if neurons.passthrough:
        for layer in neurons.layers:
            encode_sdr(layer.values, cortex.sdr.add())
    else:
        encode_sdr(neurons.values, cortex.sdr.add())

    return cortex


def encode_sdr(arr, out=None):
    sdr = out or sdr_pb2.SparseDistributedRepresentation()
    sdr.shape.extend(arr.shape)

    on_bits = np.flatnonzero(arr)
    sdr.on_bits.extend(on_bits)

    return sdr
