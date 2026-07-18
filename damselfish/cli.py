from __future__ import annotations

import argparse
import logging
import os

import uvicorn

from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Damselfish intelligent model router")
    parser.add_argument("--config", default=os.environ.get("DAMSELFISH_CONFIG", "config.yml"))
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()
    config = load_config(args.config)
    os.environ["DAMSELFISH_CONFIG"] = args.config
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )
    uvicorn.run(
        "damselfish.app:build_default_app",
        factory=True,
        host=args.host or config.host,
        port=args.port or config.port,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
