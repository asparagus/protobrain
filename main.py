#!/usr/bin/python
# -*- coding: utf-8 -*-
import time
from protobrain.proto import experiment_pb2
from protobrain import scientist


if __name__ == '__main__':
    # TODO(ariel): these should come from stdin
    exp = experiment_pb2.Experiment()
    exp.live = False
    exp.sensor.representation_size = 25
    exp.cortex.layer.extend([40, 40, 30, 20])
    exp.iters = 3

    print(exp)
    #############################
    sc = scientist.Scientist()
    res = sc.run(exp)

    print(res)
