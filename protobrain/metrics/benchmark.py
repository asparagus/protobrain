#!/usr/bin/python
# -*- coding: utf-8 -*-
from protobrain.benchmark import metric


class Benchmark(object):
    def __init__(self, metrics):
        self.metrics = metrics

    def run(self, brains, inputs, verbose=False):
        metric_results = []
        for i, brain in enumerate(brains):
            brain_results = {}
            for inp in inputs:
                brain.sensor.feed(inp)
                brain.compute()
                for metric in self.metrics:
                    metric.next(brain.neurons)

            for metric in self.metrics:
                result = metric.compute()
                brain_results[metric.name] = result
                if verbose:
                    print('Brain #%i - %s' % (i, result))

            metric_results.append(brain_results)
        return metric_results
