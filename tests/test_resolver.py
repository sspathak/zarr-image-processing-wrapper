import pytest
from chunkmirage.pipeline import PipelineSpec
from zarr_proxy.resolver import (
    convert_legacy_config,
    decode_payload,
    dynamic_pipeline_resolver,
    encode_payload,
)


def test_encode_and_decode_plain():
    data = {"source": "s3://bucket/test.zarr", "ops": [{"op": "frangi", "sigmas": [1.0, 2.0]}]}
    payload = encode_payload(data, compress=False)
    decoded = decode_payload(payload)
    assert decoded == data


def test_encode_and_decode_deflate():
    data = {
        "source": "s3://bucket/test.zarr",
        "ops": [{"op": "gaussian", "sigma": 2.0}, {"op": "threshold", "low": 50}],
    }
    # Matches ngplatform deflate-raw compression
    payload = encode_payload(data, compress=True)
    decoded = decode_payload(payload)
    assert decoded == data


def test_convert_legacy_config():
    legacy = {
        "src_url": "s3://my-bucket/dataset.zarr",
        "method": "frangi",
        "params": {"sigmas": [1.0, 2.0], "black_ridges": False},
        "target_dtype": "uint8",
    }
    spec_dict = convert_legacy_config(legacy)
    assert spec_dict["source"] == "s3://my-bucket/dataset.zarr"
    assert len(spec_dict["ops"]) == 3
    assert spec_dict["ops"][0]["op"] == "frangi"
    assert spec_dict["ops"][0]["sigmas"] == [1.0, 2.0]
    assert spec_dict["ops"][1]["op"] == "scale"
    assert spec_dict["ops"][1]["factor"] == 255.0
    assert spec_dict["ops"][2]["op"] == "cast"
    assert spec_dict["ops"][2]["dtype"] == "uint8"


def test_dynamic_resolver():
    # 1. Test resolving standard PipelineSpec payload
    spec_data = {
        "source": "s3://bucket/data.zarr",
        "ops": [{"op": "sato", "sigmas": [1.5]}],
    }
    payload = encode_payload(spec_data)
    resolved = dynamic_pipeline_resolver(payload)
    assert isinstance(resolved, PipelineSpec)
    assert resolved.source == "s3://bucket/data.zarr"
    assert len(resolved.ops) == 1
    assert resolved.ops[0]["op"] == "sato"

    # 2. Test resolving legacy config payload
    legacy_data = {
        "src_url": "s3://bucket/legacy.zarr",
        "method": "otsu",
        "params": {"value": 1},
    }
    legacy_payload = encode_payload(legacy_data)
    resolved_legacy = dynamic_pipeline_resolver(legacy_payload)
    assert isinstance(resolved_legacy, PipelineSpec)
    assert resolved_legacy.source == "s3://bucket/legacy.zarr"

    # 3. Test non-payload name returns None
    assert dynamic_pipeline_resolver("not_a_payload") is None
