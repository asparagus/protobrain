#!/usr/bin/python
# -*- coding: utf-8 -*-
# from brain.sensor import SensorGroup
import numpy as np


class Brain:
    """A class for handling a brain."""

    def __init__(self, neocortex, sensors):
        self._neocortex = neocortex
        self._sensor = sensors  # Hopefully it's just one
        # self._sensor = SensorGroup(sensors)
        #     if isinstance(sensors, list) else sensor

        self._connect_input()

    def _connect_input(self):
        """Connect the Neocortex to a sensor."""
        for minicolumn in self._neocortex._layers[0]:
            selected_synaptic_indices = np.random.choice(
                self.num_inputs,
                size=minicolumn.num_synapses,
                replace=False
            )

            self._sensor.emit.subscribe(
                lambda values: minicolumn._set_input(
                    values[selected_synaptic_indices]
                )
            )

    @property
    def num_inputs(self):
        """The number of inputs the brain can receive through its sensors."""
        return self._sensor.num_values
