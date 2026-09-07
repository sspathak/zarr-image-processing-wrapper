import json
import urllib.parse
from zarr_proxy.config import ProxyConfig, encode_config

src = 'https://storage.googleapis.com/ucl-hip-ct-35a68e99feaae8932b1d44da0358940b/LADAF-2021-17/heart/8.01um_complete-organ_bm18.ome.zarr'

def make_link(cfg, name, shader_range=[0, 255]):
    payload = encode_config(cfg)
    source_url = f'zarr3://http://128.40.89.2:8000/{payload}'
    state = {
        'dimensions': {'x': [8.01e-06, 'm'], 'y': [8.01e-06, 'm'], 'z': [8.01e-06, 'm']},
        'position': [5928, 6807.5, 9495.5],
        'crossSectionScale': 10,
        'projectionOrientation': [0.3094263970851898, 0.20628425478935242, 0, -0.9282791018486023],
        'projectionScale': 4096,
        'layers': [{
            'type': 'image',
            'source': source_url,
            'tab': 'source',
            'opacity': 1,
            'shaderControls': {'normalized': {'range': shader_range}},
            'volumeRendering': 'max',
            'volumeRenderingDepthSamples': 256,
            'name': name
        }],
        'showSlices': False,
        'selectedLayer': {'visible': True, 'layer': name},
        'layout': '4panel'
    }
    encoded = urllib.parse.quote(json.dumps(state, separators=(',', ':')))
    return f'http://128.40.89.2:8000/viewer/#!{encoded}'

print("=== 1. Ultra-Fast 3D Gaussian Smoothing (sigma=1.2) ===")
cfg_gauss = ProxyConfig(src_url=src, method='gaussian', params={'sigma': 1.2, 'vmin': 18000, 'vmax': 26000}, target_dtype='uint8', normalize=True)
print(make_link(cfg_gauss, 'gaussian-denoise-1.2'))

print("\n=== 2. Fast Single-Scale Frangi (sigma=1.5) ===")
cfg_fast_frangi = ProxyConfig(src_url=src, method='frangi', params={'sigmas': [1.5], 'alpha': 0.5, 'beta': 0.5, 'black_ridges': False, 'vmin': 18000, 'vmax': 26000}, target_dtype='uint8', normalize=True)
print(make_link(cfg_fast_frangi, 'fast-frangi-1.5'))

print("\n=== 3. Fast Sato Tubeness (sigmas=[1.0, 2.0]) ===")
cfg_sato = ProxyConfig(src_url=src, method='sato', params={'sigmas': [1.0, 2.0], 'black_ridges': False, 'vmin': 18000, 'vmax': 26000}, target_dtype='uint8', normalize=True)
print(make_link(cfg_sato, 'sato-tubeness-1-2'))

print("\n=== 4. Fast Otsu Threshold Segmentation ===")
cfg_otsu = ProxyConfig(src_url=src, method='threshold_otsu', params={'vmin': 18000, 'vmax': 26000}, target_dtype='uint8', normalize=True)
print(make_link(cfg_otsu, 'otsu-threshold'))
