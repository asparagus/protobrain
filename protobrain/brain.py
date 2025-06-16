"""The top-level module for dealing with a brain instance."""

from protobrain import computation as _computation
from protobrain import learning as _learning
from protobrain import neuron
from protobrain import sensor


class Brain:
    """A class for handling a protobrain."""

    def __init__(
        self,
        neurons: neuron.Neurons,
        sensor: sensor.Sensor,
        computation: _computation.Computation | None = None,
        learning: _learning.Learning | None = None,
    ):
        """Initialize the protobrain.

        Args:
            neurons: Neuronal architecture
            sensor: Sensor that feeds inputs to the brain
            computation: Optional - computation function that the neurons use.
                Overrides any computation function that has been previously set
                by these neurons.
            learning: Optional - learning function that the neurons use.
                Overrides any learning function that has been previously set by
                these neurons.
        """
        self.neurons = neurons
        self.sensor = sensor
        self.computation = computation
        self.learning = learning

        neurons.input = sensor

    def compute(self):
        """Compute the next brain state."""
        return self.neurons.compute()

    def learn(self):
        """Learn and adapt connections."""
        self.neurons.learn()

    @property
    def computation(self):
        return self.neurons.computation

    @computation.setter
    def computation(self, value):
        self.neurons.computation = value

    @property
    def learning(self):
        return self.neurons.learning

    @learning.setter
    def learning(self, value):
        self.neurons.learning = value
