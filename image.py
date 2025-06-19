"""Visualizations for image encoder."""

import io

import numpy as np
import plotly.express as px
import streamlit as st
from PIL import Image

from protobrain.encoders.image import BlackWhiteEncoder


DEFAULT_WIDTH = 30
DEFAULT_HEIGHT = 30
DEFAULT_MIN_RESOLUTION = 4
DEFAULT_BRIGHTNESS_BUCKETS = 2


def display_encoding(values: np.array):
    length = values.shape[0]
    sqrt = int(np.ceil(np.sqrt(length)))
    missing = sqrt * sqrt - length
    expanded = np.concatenate([values, np.zeros(missing)])
    formatted = np.astype(expanded, np.uint8).reshape((sqrt, sqrt))
    return px.imshow(formatted)


left, right = st.columns(2)
left_result: np.ndarray | None = None
right_result: np.ndarray | None = None

left_image = left.file_uploader("Left Image", label_visibility="visible")
if left_image:
    bytes_data = left_image.getvalue()
    with left.popover("Original image"):
        st.image(bytes_data)
right_image = right.file_uploader("Right Image", label_visibility="visible")
if right_image:
    bytes_data = right_image.getvalue()
    with right.popover("Original image"):
        st.image(bytes_data)


left, right = st.columns(2)
width = left.number_input(
    label="Width",
    min_value=1,
    max_value=100,
    value=DEFAULT_WIDTH,
)
height = right.number_input(
    label="Height",
    min_value=1,
    max_value=100,
    value=DEFAULT_HEIGHT,
)


min_spatial_resolution = st.number_input(
    label="Min. Spatial Resolution",
    min_value=1,
    max_value=min(DEFAULT_HEIGHT, DEFAULT_WIDTH),
    value=min(width, height, DEFAULT_MIN_RESOLUTION),
)

brightness_buckets = st.number_input(
    label="Brightness Buckets",
    min_value=2,
    max_value=10,
    value=DEFAULT_BRIGHTNESS_BUCKETS,
)

normalize = st.checkbox(label="Normalize")

encoder = BlackWhiteEncoder(
    height=height,
    width=width,
    min_spatial_resolution=min_spatial_resolution,
    brightness_buckets=brightness_buckets,
    normalize=normalize,
)

left, right = st.columns(2)
if left_image:
    actual_image = Image.open(io.BytesIO(left_image.getvalue())).resize((height, width))
    bw_image = actual_image.convert("L")
    left.plotly_chart(px.imshow(np.asarray(actual_image)))
    left.plotly_chart(px.imshow(np.asarray(bw_image)))
    left_result = encoder.encode(np.asarray(bw_image))
    left.plotly_chart(display_encoding(left_result))
    left.info(f"{len(left_result)}-dimensional vector")

if right_image:
    actual_image = Image.open(io.BytesIO(right_image.getvalue())).resize(
        (height, width)
    )
    bw_image = actual_image.convert("L")
    right.plotly_chart(px.imshow(np.asarray(actual_image)))
    right.plotly_chart(px.imshow(np.asarray(bw_image)))
    right_result = encoder.encode(np.asarray(bw_image))
    right.plotly_chart(display_encoding(right_result))
    right.info(f"{len(right_result)}-dimensional vector")


if left_result is not None and right_result is not None:
    similarity = sum(left_result == right_result) / left_result.shape[0]
    st.success(f"Similarity: {similarity:.2f}", icon="✨")
