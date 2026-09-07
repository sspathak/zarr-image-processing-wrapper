import pytest
from zarr_proxy.config import ProxyConfig, decode_config, encode_config


def test_config_roundtrip():
    cfg = ProxyConfig(
        src_url="s3://test-bucket/data.zarr",
        method="frangi",
        params={"sigmas": [1.0, 3.0], "alpha": 0.5},
        halo=8,
        target_dtype="uint8",
    )
    payload = encode_config(cfg)
    assert isinstance(payload, str)

    decoded = decode_config(payload)
    assert decoded.src_url == "s3://test-bucket/data.zarr"
    assert decoded.method == "frangi"
    assert decoded.params == {"sigmas": [1.0, 3.0], "alpha": 0.5}
    assert decoded.halo == 8
    assert decoded.target_dtype == "uint8"


def test_config_padding_variations():
    cfg = ProxyConfig(src_url="s3://bucket/path.zarr", method="sato")
    payload = encode_config(cfg)

    # Decode without padding
    d1 = decode_config(payload.rstrip("="))
    assert d1.src_url == "s3://bucket/path.zarr"
    assert d1.method == "sato"

    # Decode with extra padding if valid
    rem = len(payload) % 4
    padded = payload + ("=" * ((4 - rem) % 4))
    d2 = decode_config(padded)
    assert d2.src_url == "s3://bucket/path.zarr"


def test_config_validation_errors():
    with pytest.raises(ValueError, match="Empty configuration payload"):
        decode_config("")

    # Invalid base64
    with pytest.raises(ValueError):
        decode_config("@@@invalid_base64@@@")

    # Valid base64 but missing src_url
    import base64, json
    missing_src = base64.urlsafe_b64encode(json.dumps({"method": "frangi"}).encode()).decode()
    with pytest.raises(ValueError, match="missing required 'src_url'"):
        decode_config(missing_src)
