"""Module for image representations."""

import itertools
import logging

import numpy as np
import torch

from protobrain import sensor


logger = logging.getLogger(__name__)


class ImageEncoder(sensor.Encoder[np.ndarray]):
    """Base encoder for image types."""

    def __init__(self, height: int, width: int, length: int):
        """Initialize the encoder.

        Args:
            height: The height of the input image
            width: The width of the input image
            length: The length of the encoded representation
            sparsity: The sparsity of the encoded representation
        """
        super().__init__(default_value=0, shape=(length,))
        self.height = height
        self.width = width
        self.length = length


class BlackWhiteEncoder(ImageEncoder):
    """Encoder for black & white images."""

    def __init__(
        self,
        height: int,
        width: int,
        min_spatial_resolution: int | None = None,
        brightness_buckets: int = 2,
        normalize: bool = True,
    ):
        min_spatial_resolution = (
            max(1, min(width, height) // 16)
            if min_spatial_resolution is None
            else min_spatial_resolution
        )
        width_fits = width // min_spatial_resolution
        height_fits = height // min_spatial_resolution
        min_fits = min(width_fits, height_fits)
        minimum_width_resolution = max(min_spatial_resolution, width // min_fits)
        minimum_height_resolution = max(min_spatial_resolution, height // min_fits)
        expected_height = (height - (minimum_height_resolution - 1) - 1) // max(
            1, minimum_height_resolution // 2
        ) + 1
        expected_width = (width - (minimum_width_resolution - 1) - 1) // max(
            1, minimum_width_resolution // 2
        ) + 1
        encoder_length = sum(
            (expected_height - min_fits + 1 + z) * (expected_width - min_fits + 1 + z)
            for z in range(min_fits)
        )
        super().__init__(
            height=height, width=width, length=encoder_length * brightness_buckets
        )
        self.min_spatial_resolution = min_spatial_resolution
        self.brightness_buckets = brightness_buckets
        self.brightness_thresholds = np.arange(0, 1, 1.0 / brightness_buckets)
        self.noise_value = 0.5 / brightness_buckets
        self.filter = torch.ones(
            1, 1, minimum_height_resolution, minimum_width_resolution
        ) * (1 / (minimum_height_resolution * minimum_width_resolution))
        self.pool_filter = torch.ones(1, 1, 2, 2) * 0.25
        self.repeats = min_fits - 1
        self.normalize = normalize

    def encode(self, value: np.ndarray):
        """Encode the value to a binary representation.

        Args:
            value: The value to encode

        Returns:
            The encoded value.
        """
        assert value.shape == (self.height, self.width), (
            f"Expected input of size {self.height} x {self.width}"
        )
        if self.normalize:
            value = value / max(1e-6, value.max())
        elif value.dtype == np.uint8:
            value = np.float32(value) / 255
        input_tensor = torch.reshape(
            torch.Tensor(value), shape=(1, 1, self.height, self.width)
        )
        _, _, filter_height, filter_width = self.filter.shape
        stride = [max(1, filter_height // 2), max(1, filter_width // 2)]
        base_results = [
            torch.nn.functional.conv2d(
                input_tensor,
                weight=self.filter,
                stride=stride,
                padding=0,
            )
        ]
        noise = torch.randn_like(base_results[0]) * self.noise_value
        base_results[0] = torch.clamp(base_results[0] + noise, min=0, max=1)
        for _ in range(self.repeats):
            base_results.append(
                torch.nn.functional.conv2d(
                    base_results[-1],
                    weight=self.pool_filter,
                    stride=1,
                )
            )
        unthresholded_results = np.concatenate(
            [r.flatten().numpy() for r in base_results]
        )

        thresholded_results = [
            (unthresholded_results <= high) & (unthresholded_results >= low)
            for low, high in itertools.pairwise(
                np.concatenate([self.brightness_thresholds, [1]])
            )
        ]
        return np.concatenate(thresholded_results)
