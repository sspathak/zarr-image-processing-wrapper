import numpy as np
import pytest
from zarr_proxy.codecs_util import decode_chunk_bytes, encode_chunk_bytes
from zarr_proxy.config import ProxyConfig
from zarr_proxy.filters import apply_filter, calculate_halo
from zarr_proxy.processor import (
    calculate_bounding_boxes,
    format_chunk_key,
    parse_chunk_key,
)


def test_calculate_halo():
    # Frangi with sigmas [1.0, 2.0] -> max sigma is 2.0 -> ceil(3 * 2.0) = 6
    h_frangi = calculate_halo("frangi", {"sigmas": [1.0, 2.0]})
    assert h_frangi == [6, 6, 6]

    # Gaussian with sigma 3.0 -> ceil(3 * 3.0) = 9
    h_gauss = calculate_halo("gaussian", {"sigma": 3.0})
    assert h_gauss == [9, 9, 9]

    # Constant filter requires 0 halo
    h_const = calculate_halo("constant", {})
    assert h_const == [0, 0, 0]

    # Explicit override
    h_override = calculate_halo("frangi", {"sigmas": [1.0, 2.0]}, user_halo=10)
    assert h_override == [10, 10, 10]


def test_parse_and_format_chunk_key():
    assert parse_chunk_key("c/0/1/2") == (0, 1, 2)
    assert parse_chunk_key("c/0/0/0") == (0, 0, 0)
    assert parse_chunk_key("0/1/2") == (0, 1, 2)
    assert parse_chunk_key("c.1.2.3", separator=".") == (1, 2, 3)

    assert format_chunk_key((0, 1, 2), "default", "/") == "c/0/1/2"
    assert format_chunk_key((0, 1, 2), "v2", "/") == "0/1/2"


def test_bounding_boxes_interior():
    # Dataset 100x100x100, chunk size 32, halo 6
    # Target chunk (1, 1, 1): covers [32, 64)
    # Desired range: [26, 70)
    # Neighbor chunks: 26//32 = 0 to (70-1)//32 = 2
    boxes = calculate_bounding_boxes(
        target_coords=(1, 1, 1),
        shape=[100, 100, 100],
        chunk_shape=[32, 32, 32],
        halo=[6, 6, 6],
    )
    assert boxes["c_min"] == [0, 0, 0]
    assert boxes["c_max"] == [2, 2, 2]
    assert boxes["pad_left"] == [0, 0, 0]
    assert boxes["pad_right"] == [0, 0, 0]


def test_bounding_boxes_corner():
    # Corner chunk (0, 0, 0)
    # Target chunk covers [0, 32)
    # Desired range: [-6, 38) -> valid [0, 38)
    # Left pad needed = 6
    boxes = calculate_bounding_boxes(
        target_coords=(0, 0, 0),
        shape=[100, 100, 100],
        chunk_shape=[32, 32, 32],
        halo=[6, 6, 6],
    )
    assert boxes["c_min"] == [0, 0, 0]
    assert boxes["c_max"] == [1, 1, 1]
    assert boxes["pad_left"] == [6, 6, 6]
    assert boxes["pad_right"] == [0, 0, 0]


def test_volumetric_filters_3d():
    vol = np.zeros((32, 32, 32), dtype=np.float32)
    vol[16, :, 16] = 10.0  # Synthetic cylinder / tube

    # Frangi filter
    frangi_out = apply_filter(
        volume=vol,
        method="frangi",
        params={"sigmas": [1.0, 2.0], "black_ridges": False},
        target_dtype="uint8",
    )
    assert frangi_out.shape == (32, 32, 32)
    assert frangi_out.dtype == np.uint8
    # Center of the tube should have high response
    assert frangi_out[16, 16, 16] > frangi_out[0, 0, 0]

    # Sato filter
    sato_out = apply_filter(
        volume=vol,
        method="sato",
        params={"sigmas": [1.0, 2.0], "black_ridges": False},
        target_dtype="uint8",
    )
    assert sato_out.shape == (32, 32, 32)
    assert sato_out.dtype == np.uint8
    assert sato_out[16, 16, 16] > sato_out[0, 0, 0]

    # Constant filter
    const_out = apply_filter(
        volume=vol,
        method="constant",
        params={"value": 77},
        target_dtype="uint8",
    )
    assert np.all(const_out == 77)


def test_halo_boundary_continuity():
    """
    Verify that filtering a chunk with halo padding matches full-volume filtering
    at interior voxels to floating-point precision, proving elimination of chunk edge artifacts.
    """
    from skimage.filters import frangi

    vol = np.zeros((64, 64, 64), dtype=np.float32)
    vol[32, :, 32] = 10.0

    full_filtered = frangi(vol, sigmas=[1.0, 2.0], black_ridges=False)

    sub = vol[0:38, 0:38, 0:38]
    sub_padded = np.pad(sub, ((6, 0), (6, 0), (6, 0)), mode="reflect")
    filt_sub = frangi(sub_padded, sigmas=[1.0, 2.0], black_ridges=False)
    cropped = filt_sub[6:38, 6:38, 6:38]

    # Compare interior region [10:32, 10:32, 10:32]
    diff = np.abs(full_filtered[10:32, 10:32, 10:32] - cropped[10:32, 10:32, 10:32])
    assert np.max(diff) < 1e-4


def test_codec_roundtrip():
    arr = np.random.randint(0, 255, size=(32, 32, 32), dtype=np.uint8)
    for comp in ["blosc", "gzip", "zstd", "none"]:
        compressed = encode_chunk_bytes(arr, compression=comp, cname="zstd", clevel=5)
        codecs_spec = [{"name": "bytes", "configuration": {"endian": "little"}}]
        if comp != "none":
            codecs_spec.append({"name": comp, "configuration": {"cname": "zstd", "level": 5, "clevel": 5}})
        decoded = decode_chunk_bytes(compressed, codecs_spec, (32, 32, 32), "uint8")
        assert np.array_equal(arr, decoded)
