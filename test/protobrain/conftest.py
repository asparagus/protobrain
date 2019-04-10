#!/usr/bin/python
# -*- coding: utf-8 -*-
import pytest
from protobrain.computation import Computation


class DummyComputation(Computation):
    def __init__(self, return_value):
        self._return_value = return_value

    def __call__(self, inputs, feedback, inhibitions):
        return self._return_value

@pytest.fixture(scope='session')
def dummy_computation():
    return DummyComputation
