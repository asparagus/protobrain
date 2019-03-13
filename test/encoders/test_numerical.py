#!/usr/bin/python
# -*- coding: utf-8 -*-
from encoders.numerical import CyclicEncoder
from encoders.numerical import FixedRangeEncoder


def test_fixed_range():
    fixed_range = FixedRangeEncoder(min=1, max=4, length=5, sparsity=0.4)
    assert all([1, 1, 0, 0, 0] == fixed_range.encode(1))
    assert all([0, 1, 1, 0, 0] == fixed_range.encode(2))
    assert all([0, 0, 1, 1, 0] == fixed_range.encode(3))
    assert all([0, 0, 0, 1, 1] == fixed_range.encode(4))

def test_cyclic():
    cyclic = CyclicEncoder(min=1, max=5, length=5, sparsity=0.4)
    assert all([1, 1, 0, 0, 0] == cyclic.encode(1))
    assert all([0, 1, 1, 0, 0] == cyclic.encode(2))
    assert all([0, 0, 1, 1, 0] == cyclic.encode(3))
    assert all([0, 0, 0, 1, 1] == cyclic.encode(4))
    assert all([1, 0, 0, 0, 1] == cyclic.encode(5))
