"""A module for defining different kinds of neuronal learning."""

import abc
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from protobrain import neuron


class Learning(abc.ABC):
    """The base class for all learning functions."""

    @abc.abstractmethod
    def __call__(self, neurons: "neuron.Neurons"):
        """Make the neurons learn.

        Args:
            neurons: The neurons to update
        """
        raise NotImplementedError()


class HebbianLearning(Learning):
    """A kind of learning that favors the connection of co-occurrent neurons.

    'Neurons that fire together, wire together.'
    'Neurons that fire apart, wire apart'
    """

    def __init__(self, increase: float = 0.05, decrease: float = 0.002):
        """Initialize the Hebbian learning with appropriate constants.

        Args:
            increase: The amount by which to increase the synapse strengths
            decrease: The amount by which to decrease the synapse strengths
        """
        self.increase = increase
        self.decrease = decrease

    def __call__(self, neurons: "neuron.Neurons"):
        """Make the neurons learn.

        Increase the weights on synapses where input and output were active.
        Decrease the weights on synapses where only the output was active.

        Args:
            neurons: The neurons to update
        """
        active_neurons = np.array(neurons.output.values) == 1
        for _, input_unit in neurons.inputs.items():
            active_inputs = np.array(input_unit.values) == 1
            zeros = input_unit.synapses == 0
            input_unit.synapses[np.ix_(active_inputs, ~active_neurons)] -= self.decrease
            input_unit.synapses[np.ix_(~active_inputs, active_neurons)] -= self.decrease
            input_unit.synapses[np.ix_(active_inputs, active_neurons)] += self.increase

            input_unit.synapses[zeros] = 0
            input_unit.synapses = np.clip(input_unit.synapses, 0, 1)
