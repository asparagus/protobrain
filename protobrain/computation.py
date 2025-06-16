"""A module for defining different kinds of neuronal computations."""

import abc
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from protobrain import synapses


class Computation(abc.ABC):
    """Base class for all other computations to inherit from."""

    @abc.abstractmethod
    def __call__(self):
        """Compute the neurons' output."""
        raise NotImplementedError()

    def __repr__(self):
        """The name of this computation."""
        return self.__class__.__name__


class StandardComputation(Computation):
    """The standard computation is a thresholded dot product."""

    def __init__(self, threshold: float):
        """Initialize the computation.

        Args:
            threshold: The cut-off threshold for the binary output
        """
        self.threshold = threshold

    def __call__(self, main: "synapses.Input") -> np.array:
        """Compute the neurons' output.

        For each neuron, adds up the weight of the active synapses,
        then applies a threshold to get the activations.

        Args:
            main: The main input

        Returns:
            Binary values from the computation
        """
        activations = np.dot(main.values, main.synapses)
        return activations > self.threshold


class SparseComputation(Computation):
    """A computation with a limited number of active units."""

    def __init__(self, n: int | float):
        """Initialize the computation.

        Args:
            n: The number of neurons to activate in each timestep. If fractional,
              will activate this fraction of neurons.
        """
        self.n = n

    def __call__(self, main: "synapses.Input"):
        """Compute the neurons' output.

        For each neuron, adds up the weight of the active synapses,
        then sets the top n to be active.

        Args:
            main: The main input

        Returns:
            Binary values from the computation
        """
        activations = np.dot(main.values, main.synapses)

        n = (
            int(np.ceil(self.n * len(activations)))
            if isinstance(self.n, float)
            else self.n
        )

        top_indices = activations.argsort()[-n:]

        result = np.zeros(len(activations))
        result[top_indices] = 1
        return result
