#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from protobrain.proto.snapshot_pb2 import SparseDistributedRepresentation


def np_to_sdr(arr, out=None):
    sdr = out or SparseDistributedRepresentation()
    sdr.dimensions.extend(arr.shape)

    on_bits = np.flatnonzero(arr)
    sdr.on_bits.extend(on_bits)

    return sdr
