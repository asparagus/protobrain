"""Tests for image encoder module."""

import pytest

import numpy as np

from protobrain.encoders.image import BlackWhiteEncoder


@pytest.mark.parametrize(
    argnames=["height", "width", "min_spatial_resolution"],
    argvalues=[
        (20, 20, 3),
        (20, 30, 4),
        (10, 10, 2),
    ],
)
def test_dimensions_correct(height: int, width: int, min_spatial_resolution: int):
    """Verify that the encoder dimensions match what it expects."""
    encoder = BlackWhiteEncoder(
        height=height, width=width, min_spatial_resolution=min_spatial_resolution
    )
    sample_data = np.random.random((height, width))
    output = encoder.encode(sample_data)
    assert len(output) == encoder.length
