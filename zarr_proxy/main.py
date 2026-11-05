"""
CLI entry point for running the virtual Zarr compute server or encoding pipeline specs.
"""

import argparse
import json
import sys
import uvicorn
from zarr_proxy.resolver import encode_payload


def run_server():
    parser = argparse.ArgumentParser(
        description="Virtual Zarr Compute Server powered by chunkmirage"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host address to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")

    subparsers = parser.add_subparsers(dest="command", help="Subcommands")

    # Subcommand: encode
    encode_parser = subparsers.add_parser("encode", help="Encode a pipeline specification into a URL payload")
    encode_parser.add_argument("--src-url", required=True, help="Source S3 / URL to Zarr dataset")
    encode_parser.add_argument("--method", default="frangi", help="Filter method (frangi, sato, gaussian, otsu, etc.)")
    encode_parser.add_argument("--sigmas", nargs="+", type=float, default=[1.0, 2.0], help="Sigmas for filter")
    encode_parser.add_argument("--target-dtype", default="uint8", help="Target output data type")
    encode_parser.add_argument("--proxy-host", default="http://localhost:8000", help="Proxy server address")
    encode_parser.add_argument(
        "--backend",
        choices=["local", "dask", "webworker", "workstation"],
        default="local",
        help="Compute backend target (local, dask, webworker, workstation)",
    )
    encode_parser.add_argument("--compress", action="store_true", help="Use ngplatform deflate-raw compression")

    args = parser.parse_args()

    if args.command == "encode":
        ops = [{"op": args.method, "sigmas": args.sigmas}]
        if args.target_dtype and args.target_dtype != "float32":
            ops.append({"op": "cast", "dtype": args.target_dtype})

        spec = {
            "source": args.src_url,
            "ops": ops,
        }
        proxy_host = args.proxy_host.rstrip("/")
        if args.backend == "webworker":
            from zarr_proxy.backends.webworker import build_ngplatform_url, to_ngplatform_pipeline

            spec = to_ngplatform_pipeline(spec, compile_ops=True)
            payload = encode_payload(spec, compress=True)
            ng_source = build_ngplatform_url(spec, base_url=f"{proxy_host}/", compile_ops=False)
        else:
            payload = encode_payload(spec, compress=args.compress)
            ng_source = f"zarr3://{proxy_host}/{payload}/zarr3"

        print("Pipeline Specification JSON:")
        print(json.dumps(spec, indent=2))
        print("\nURL-safe Payload:")
        print(payload)
        print("\nNeuroglancer Layer Source URL:")
        print(ng_source)
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
