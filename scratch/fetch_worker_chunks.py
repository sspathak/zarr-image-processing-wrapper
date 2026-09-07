import os
import re
import urllib.request

base_url = "https://neuroglancer-demo.appspot.com"
dest_dir = "zarr_proxy/static/neuroglancer"

def download_file(fname):
    target = os.path.join(dest_dir, fname)
    if os.path.exists(target):
        return
    url = f"{base_url}/{fname}"
    try:
        print(f"Downloading {url}...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
        with open(target, "wb") as f:
            f.write(data)
        print(f"Saved {fname} ({len(data)} bytes)")
    except Exception as e:
        print(f"Failed to fetch {fname}: {e}")

# Specifically download the worker chunk requested
download_file("34.f9e32e7c95411506.js")

# Find all chunk references across all .js files in dest_dir
pattern = re.compile(r'["\']([0-9a-zA-Z_.-]+\.(?:js|wasm))["\']')
all_files = set(os.listdir(dest_dir))
to_scan = [f for f in all_files if f.endswith(".js")]

for js_file in to_scan:
    path = os.path.join(dest_dir, js_file)
    with open(path, "r", errors="ignore") as f:
        content = f.read()
    for ref in pattern.findall(content):
        if not ref.startswith("http") and not ref.startswith("//") and not os.path.exists(os.path.join(dest_dir, ref)):
            download_file(ref)

print("Finished syncing viewer assets.")
