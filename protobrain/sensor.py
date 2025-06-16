"""Module for handling a sensory layer."""

import abc

import numpy as np

from protobrain import synapses


class Encoder[T](abc.ABC):
    """Base class for encoders.

    Encoders can take in a value and convert it to a representative binary
    array suitable for a Sensor.
    """

    def __init__(self, default_value: T, shape: tuple[int, ...]):
        """Initialize the encoder.

        Args:
            default_value: The starting value
            shape: The output shape
        """
        self.default_value = default_value
        self.shape = shape

    @abc.abstractmethod
    def encode(self, value: T):
        """Encode the value to a binary representation.

        Args:
            value: The value to encode

        Returns:
            The encoded value.
        """
        raise NotImplementedError()


class Sensor[T]:
    """A class for handling input data."""

    def __init__(self, encoder: Encoder[T]):
        """Initialize the Sensor."""
        self._encoder = encoder
        self._value = encoder.default_value
        self.output = synapses.Output(encoder.shape)

    def feed(self, value: T):
        """Feed a value to the sensor."""
        self._value = value
        self.output.values = self._encoder.encode(value)

    @property
    def value(self) -> T:
        """Get the value encoded by this sensor."""
        return self._value

    @value.setter
    def value(self, output: T):
        """Set the value encoded by this sensor."""
        self.feed(output)

    @property
    def values(self) -> np.array:
        """The values from the output unit."""
        return self.output.values

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of this sensor's outputs."""
        return self.output.shape
