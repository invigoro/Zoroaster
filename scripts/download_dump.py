"""Download a Wikimedia dump file, respecting Wikimedia's User-Agent policy.

Downloads go to `<destination>.part` and are renamed only once complete, and
checked against the server's Content-Length when it sends one. So an
interrupted download never leaves a truncated file that looks finished.

Usage:
    python scripts/download_dump.py <url> <destination_path>
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests

from src.common import USER_AGENT


def download(url: str, destination: Path, chunk_size: int = 1 << 20, progress: bool = True) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    with requests.get(
        url, headers={"User-Agent": USER_AGENT}, stream=True, timeout=30
    ) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", 0))
        written = 0
        with open(partial, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                written += len(chunk)
                if total and progress:
                    pct = written / total * 100
                    print(
                        f"\r{destination.name}: {written / 1e6:,.1f}MB / "
                        f"{total / 1e6:,.1f}MB ({pct:.1f}%)",
                        end="",
                    )
    if progress:
        print()
    if total and written != total:
        raise IOError(f"{url}: got {written:,} bytes, expected {total:,}; kept {partial}")
    os.replace(partial, destination)


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 2:
        print(
            "Usage: python scripts/download_dump.py <url> <destination_path>",
            file=sys.stderr,
        )
        return 1
    url, dest = argv
    download(url, Path(dest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
