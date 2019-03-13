#!/usr/bin/python
# -*- coding: utf-8 -*-
import pytest
import numpy as np
from brain.event import EventVerifier
from brain.sensor import Sensor


def test_wrong_length():
    sensor = Sensor(10)
    with pytest.raises(ValueError):
        sensor.set_values(np.zeros(3))

def test_emit():
    sensor = Sensor(10)
    verify = EventVerifier(sensor.emit)

    sensor.set_values(np.zeros(10))

    assert verify.has_run
