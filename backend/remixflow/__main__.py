"""CLI entrypoint: ``python -m remixflow serve`` / ``remixflow serve``."""

from __future__ import annotations

import argparse

from . import __version__


def main() -> None:
    parser = argparse.ArgumentParser(prog="remixflow", description="RemixFlow server")
    parser.add_argument("--version", action="version", version=f"remixflow {__version__}")
    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="Run the API + UI server")
    serve.add_argument("--host", default="127.0.0.1")
    # 8770 default: this host runs other services on 8000 and 8188 (llama.cpp).
    serve.add_argument("--port", type=int, default=8770)
    serve.add_argument("--reload", action="store_true", help="Auto-reload (dev)")

    args = parser.parse_args()
    if args.command in (None, "serve"):
        import uvicorn

        host = getattr(args, "host", "127.0.0.1")
        port = getattr(args, "port", 8000)
        reload = getattr(args, "reload", False)
        # Import string form enables --reload.
        uvicorn.run("remixflow.server:app", host=host, port=port, reload=reload,
                    factory=False)
    else:  # pragma: no cover
        parser.print_help()


if __name__ == "__main__":
    main()
