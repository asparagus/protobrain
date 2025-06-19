"""Module for handling neuron connections."""

import logging
from typing import Callable

import numpy as np


log = logging.getLogger(__name__)


class Input:
    """An input connection with synapses.

    Inputs are identified by a name and set up their synapses when connected
    to an output.
    """

    def __init__(self, name: str, shape: tuple[int, ...] | int):
        """Initialize the input.

        Args:
            name: The name of the input
            shape: The shape of the neurons this input is feeding
        """
        if isinstance(shape, int):
            shape = (shape,)
        self.name = name
        self.shape = shape
        self.synapses = None
        self._connected_output = None

    @property
    def connected(self) -> bool:
        """Whether the input is connected to an output."""
        return self._connected_output is not None

    def connect(
        self,
        output: "Output",
        synapse_function: Callable[[tuple[int, ...], tuple[int, ...]], None]
        | None = None,
    ):
        """Connect this input to an output.

        Creates the synapses to match the output's dimensions.

        Args:
            output: The output to connect to
        """
        if self._connected_output is output:
            log.warning("Skipping reconnection of input-output pair")
            return  # Skip

        if synapse_function is None:
            synapse_function = Input._create_synapses

        self.synapses = synapse_function(output.shape, self.shape)
        self._connected_output = output

    @property
    def values(self) -> np.ndarray:
        """The values available to this input.

        These are taken from the connected output."""
        if not self._connected_output:
            raise IndexError(f"No input set for {self}")
        return self._connected_output.values

    @classmethod
    def _create_synapses(
        cls,
        output_shape: tuple[int, ...],
        input_shape: tuple[int, ...],
        symmetric: bool = False,
    ) -> np.ndarray:
        """Create the synapses between an input and an output.

        Args:
            output_shape: The shape of the output
            input_shape: The shape of the input
            symmetric: Whether to enforce symmetric weights

        Returns:
            A numpy tensor with the right shape and random weights
        """
        shape = output_shape + input_shape
        strength = np.random.uniform(0, 1, shape)

        return (strength + strength.T) / 2 if symmetric else strength


class Output:
    """An output to which an input can connect."""

    def __init__(self, shape: tuple[int, ...] | int):
        """Initialize the output.

        Args:
            shape: The output's shape
        """
        if isinstance(shape, int):
            shape = (shape,)
        self._values = np.zeros(shape)
        self.shape = self._values.shape

    @property
    def values(self) -> np.ndarray:
        """The values available at this output."""
        return self._values

    @values.setter
    def values(self, vals: np.ndarray) -> None:
        """Set the values on the output, verifying the shape is right."""
        if vals.shape != self.shape:
            raise ValueError(
                "Dimension mismatch when specifying output values. "
                "Expected {0}, but got {1}".format(self.shape, vals.shape)
            )
        self._values = vals

    def __getitem__(self, idxs) -> "OutputSlice":
        """Slice the Output."""
        if isinstance(idxs, slice) or isinstance(idxs, tuple):
            return OutputSlice(self, idxs)
        elif isinstance(idxs, int):
            return OutputSlice(self, slice(idxs))
        else:
            raise IndexError("Invalid slicing of an Output: {0}".format(idxs))


class OutputMerge(Output):
    """Output that merges two or more outputs."""

    def __init__(self, *outputs: Output, axis: int | None = None):
        """Initialize the output.

        Args:
            outputs: Outputs to merge
            axis: Axis along which to concatenate them
        """
        if not outputs:
            raise ValueError("Need at least two outputs to merge")

        if axis is None:
            axis = self.pick_axis(outputs)

        self._axis = axis
        self._outputs = outputs
        self.shape = np.concatenate(
            [output.values for output in self._outputs], axis=axis
        ).shape

    def _merge(self, outputs: tuple[Output], axis: int) -> np.ndarray:
        """Merge the output values.

        Args:
            outputs: The outputs to merge
            axis: The axis along which to concatenate
        """
        return np.concatenate([output.values for output in outputs], axis=axis)

    def pick_axis(self, outputs: tuple[Output]) -> int:
        """Pick an axis for concatenating the outputs.

        Args:
            outputs: The outputs that need to be merged

        Returns:
            An axis along which they can be concatenated

        Throws:
            ValueError if no axis allows concatenation
        """
        axis_options = range(len(outputs[0].shape))
        for axis in axis_options:
            try:
                self._merge(outputs, axis)
                return axis
            except Exception:
                pass
        raise ValueError(
            "No single axis can be used to merge outputs of shapes {}".format(
                [output.shape for output in outputs]
            )
        )

    @property
    def values(self) -> np.ndarray:
        """Concatenate the values from the merged outputs."""
        return self._merge(self._outputs, self._axis)


class OutputSlice(Output):
    """A slice of an output."""

    def __init__(self, output: Output, slice: slice):
        """Initialize the output.

        Args:
            output: The original output
            slice: A slice object representing the part of the output to take.
        """
        self._output = output
        self._slice = slice

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the output."""
        return self.values.shape

    @property
    def values(self) -> np.ndarray:
        """Slice the values from the internal output."""
        return self._output.values[self._slice]
