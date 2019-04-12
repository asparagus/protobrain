#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Module for handling neuron connections."""
import numpy as np


class Input(object):
    def __init__(self, name, shape):
        self.name = name
        self.shape = np.zeros(shape).shape  # TODO: Not the most efficient
        self.synapses = None
        self._connected_output = None

    @property
    def connected(self):
        return self._connected_output is not None

    # TODO: Allow providing the function for synapse creation
    def connect(self, output):
        if self._connected_output is output:
            return  # Skip

        self.synapses = Input._create_synapses(self.shape, output.shape)
        self._connected_output = output

    @property
    def values(self):
        if not self._connected_output:
            raise IndexError('No input set for {1}'.format(repr(self)))
        return self._connected_output.values

    @classmethod
    def _create_synapses(cls, input_shape, output_shape, symmetric=False):
        shape = input_shape + output_shape
        strength = np.random.uniform(0, 1, shape)

        return (
            (strength + strength.T) / 2
            if symmetric
            else strength
        )


class Output(object):
    def __init__(self, shape):
        self._values = np.zeros(shape)
        self.shape = self._values.shape

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, vals):
        """Set the values on the output. Verifies the shape"""
        if vals.shape != self.shape:
            raise ValueError(
                'Dimension mismatch when specifying output values. '
                'Expected {0}, but got {1}'.format(self.shape, vals.shape)
            )
        self._values = vals

    def __getitem__(self, idxs):
        """Slice the Output."""
        if isinstance(idxs, slice) or isinstance(idxs, tuple):
            return OutputSlice(self, idxs)
        elif isinstance(idxs, int):
            return OutputSlice(self, slice(idxs))
        else:
            raise IndexError('Invalid slicing of an Output: {0}'.format(idxs))


class OutputMerge(Output):
    def __init__(self, outputs, axis=None):
        if not outputs:
            raise ValueError('Need at least two outputs to merge')

        if axis is None:
            axis = self.pick_axis(outputs)

        self._axis = axis
        self._outputs = outputs
        self.shape = np.concatenate(
            [output.values for output in self._outputs],
            axis=axis
        ).shape

    def merge(self, outputs, axis):
        return np.concatenate(
            [output.values for output in outputs],
            axis=axis
        )

    def pick_axis(self, outputs):
        axis_options = range(len(outputs[0].shape))
        for axis in axis_options:
            try:
                self.merge(outputs, axis)
                return axis
            except:
                pass
        raise ValueError(
            'No single axis can be used to merge outputs of shapes {}'.format(
                [output.shape for output in outputs]
            ))

    @property
    def values(self):
        """Concatenate the values from the merged outputs."""
        return self.merge(self._outputs, self._axis)


class OutputSlice(Output):
    def __init__(self, output, slice):
        self._output = output
        self._slice = slice

    @property
    def shape(self):
        return self.values.shape

    @property
    def values(self):
        """Slice the values from the internal output."""
        return self._output.values[self._slice]
