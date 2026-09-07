import json
import urllib.parse
from zarr_proxy.config import ProxyConfig, encode_config

def get_urls():
    # 1. Passthrough (identity) with uint8 windowed normalization
    cfg_identity_uint8 = ProxyConfig(
        src_url='https://storage.googleapis.com/ucl-hip-ct-35a68e99feaae8932b1d44da0358940b/LADAF-2021-17/heart/8.01um_complete-organ_bm18.ome.zarr',
        method='identity',
        params={'vmin': 18000, 'vmax': 26000},
        halo=None,
        target_dtype='uint8',
        normalize=True,
        compression='blosc',
        cname='zstd',
        clevel=5
    )
    payload_id_u8 = encode_config(cfg_identity_uint8)
    
    # 2. Raw Passthrough (uint16 without normalization, original CT values)
    cfg_raw_u16 = ProxyConfig(
        src_url='https://storage.googleapis.com/ucl-hip-ct-35a68e99feaae8932b1d44da0358940b/LADAF-2021-17/heart/8.01um_complete-organ_bm18.ome.zarr',
        method='identity',
        params={},
        halo=None,
        target_dtype='uint16',
        normalize=False,
        compression='blosc',
        cname='zstd',
        clevel=5
    )
    payload_raw_u16 = encode_config(cfg_raw_u16)

    # 3. Frangi Tubeness Filter (sigmas 1.0, 2.0)
    cfg_frangi = ProxyConfig(
        src_url='https://storage.googleapis.com/ucl-hip-ct-35a68e99feaae8932b1d44da0358940b/LADAF-2021-17/heart/8.01um_complete-organ_bm18.ome.zarr',
        method='frangi',
        params={'sigmas': [1.0, 2.0], 'black_ridges': False},
        halo=None,
        target_dtype='uint8',
        normalize=True,
        compression='blosc',
        cname='zstd',
        clevel=5
    )
    payload_frangi = encode_config(cfg_frangi)

    def make_neuroglancer_url(source_url, shader_range, name):
        state = {
            "dimensions": {
                "x": [0.00000801, "m"],
                "y": [0.00000801, "m"],
                "z": [0.00000801, "m"]
            },
            "position": [5928, 6807.5, 9495.5],
            "crossSectionScale": 10,
            "projectionOrientation": [0.3094263970851898, 0.20628425478935242, 0, -0.9282791018486023],
            "projectionScale": 4096,
            "layers": [
                {
                    "type": "image",
                    "source": source_url,
                    "tab": "source",
                    "opacity": 1,
                    "shaderControls": {
                        "normalized": {
                            "range": shader_range
                        }
                    },
                    "volumeRendering": "max",
                    "volumeRenderingDepthSamples": 256,
                    "name": name
                }
            ],
            "showSlices": False,
            "selectedLayer": {
                "visible": True,
                "layer": name
            },
            "layout": "4panel"
        }
        encoded = urllib.parse.quote(json.dumps(state, separators=(',', ':')))
        return f"http://128.40.89.2:8000/viewer/#!{encoded}"

    url_u8 = make_neuroglancer_url(f"zarr3://http://128.40.89.2:8000/{payload_id_u8}", [0, 255], "heart-passthrough-uint8")
    url_u16 = make_neuroglancer_url(f"zarr3://http://128.40.89.2:8000/{payload_raw_u16}", [18192, 29950], "heart-raw-uint16")
    url_frangi = make_neuroglancer_url(f"zarr3://http://128.40.89.2:8000/{payload_frangi}", [0, 255], "heart-frangi-tubeness")

    print("--- 1. Passthrough (uint8 normalized [0, 255]) ---")
    print(url_u8)
    print("\n--- 2. Passthrough (uint16 raw [18192, 29950]) ---")
    print(url_u16)
    print("\n--- 3. Frangi Tubeness Filter ---")
    print(url_frangi)

if __name__ == "__main__":
    get_urls()
