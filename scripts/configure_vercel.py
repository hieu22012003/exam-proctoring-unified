"""Write the Vercel static-site and same-origin API proxy configuration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("backend_url", help="Public HTTPS origin of the Railway API, e.g. https://app.up.railway.app")
    args = parser.parse_args()
    parsed = urlsplit(args.backend_url.rstrip("/"))
    if (parsed.scheme != "https" or not parsed.netloc or parsed.path or parsed.query
            or parsed.fragment or parsed.username or parsed.password):
        parser.error("backend_url must be an HTTPS origin without a path, query, or credentials")

    config = {
        "$schema": "https://openapi.vercel.sh/vercel.json",
        "framework": None,
        "outputDirectory": "web",
        "rewrites": [
            {"source": "/api/:path*", "destination": f"{parsed.scheme}://{parsed.netloc}/api/:path*"},
            {"source": "/static/:path*", "destination": "/:path*"},
        ],
    }
    destination = ROOT / "vercel.json"
    destination.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
