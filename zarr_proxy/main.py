"""
CLI entry point for running the proxy server or encoding config payloads.
"""

import argparse
import sys
import uvicorn
from zarr_proxy.config import ProxyConfig, encode_config


def run_server():
    parser = argparse.ArgumentParser(
        description="Zarr v3 Neuroglancer Volumetric Filtering Proxy Server"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host address to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")

    subparsers = parser.add_subparsers(dest="command", help="Subcommands")

    # Subcommand: encode
    encode_parser = subparsers.add_parser("encode", help="Encode a config payload for Neuroglancer")
    encode_parser.add_argument("--src-url", required=True, help="Source S3 / URL to Zarr dataset")
    encode_parser.add_argument("--method", default="frangi", help="Filter method (frangi, sato, gaussian, etc.)")
    encode_parser.add_argument("--sigmas", nargs="+", type=float, default=[1.0, 2.0], help="Sigmas for filter")
    encode_parser.add_argument("--halo", type=int, default=None, help="Explicit halo radius in voxels")
    encode_parser.add_argument("--target-dtype", default="uint8", help="Target output data type")
    encode_parser.add_argument("--proxy-host", default="http://128.40.89.2:8000", help="Proxy server address (default: http://128.40.89.2:8000)")

    args = parser.parse_args()

    if args.command == "encode":
        cfg = ProxyConfig(
            src_url=args.src_url,
            method=args.method,
            params={"sigmas": args.sigmas},
            halo=args.halo,
            target_dtype=args.target_dtype,
        )
        payload = encode_config(cfg)
        proxy_host = args.proxy_host.rstrip("/")
        print("Configuration Payload:")
        print(payload)
        print("\nNeuroglancer Layer Source URL:")
        print(f"zarr3://{proxy_host}/{payload}")
        sys.exit(0)

    uvicorn.run(
        "zarr_proxy.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
    )


if __name__ == "__main__":
    run_server()
