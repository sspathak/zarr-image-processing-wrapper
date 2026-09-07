import json
import pytest
from zarr_proxy.config import ProxyConfig
from zarr_proxy.metadata import extract_array_info, falsify_zarr_metadata


def test_falsify_array_metadata():
    original = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [100, 200, 300],
        "data_type": "float32",
        "chunk_grid": {
            "name": "regular",
            "configuration": {
                "chunk_shape": [32, 32, 32]
            }
        },
        "chunk_key_encoding": {
            "name": "default",
            "configuration": {
                "separator": "/"
            }
        },
        "fill_value": 0.0,
        "codecs": [
            {
                "name": "bytes",
                "configuration": {"endian": "little"}
            },
            {
                "name": "blosc",
                "configuration": {
                    "cname": "zstd",
                    "clevel": 5,
                    "shuffle": "shuffle",
                    "typesize": 4,
                    "blocksize": 0
                }
            }
        ],
        "attributes": {}
    }
    raw_bytes = json.dumps(original).encode()
    cfg = ProxyConfig(
        src_url="s3://b/d.zarr",
        method="frangi",
        target_dtype="uint8",
        compression="blosc",
        cname="zstd",
        clevel=5,
    )

    falsified, orig = falsify_zarr_metadata(raw_bytes, cfg)

    # Verify data_type changed
    assert falsified["data_type"] == "uint8"
    assert falsified["shape"] == [100, 200, 300]
    assert falsified["chunk_grid"]["configuration"]["chunk_shape"] == [32, 32, 32]

    # Verify codecs modified for uint8 (typesize 1)
    blosc_codec = next(c for c in falsified["codecs"] if c["name"] == "blosc")
    assert blosc_codec["configuration"]["typesize"] == 1
    assert blosc_codec["configuration"]["cname"] == "zstd"

    info = extract_array_info(orig)
    assert info["shape"] == [100, 200, 300]
    assert info["chunk_shape"] == [32, 32, 32]
    assert info["key_encoding_name"] == "default"
    assert info["key_separator"] == "/"


def test_group_metadata_passthrough():
    group_meta = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {
            "ome": {"version": "0.5"}
        }
    }
    raw_bytes = json.dumps(group_meta).encode()
    cfg = ProxyConfig(src_url="s3://b/d.zarr")

    falsified, orig = falsify_zarr_metadata(raw_bytes, cfg)
    assert falsified["node_type"] == "group"
    assert falsified["attributes"]["ome"]["version"] == "0.5"
