#!/usr/bin/env python3
"""Remove macOS ._* metadata sidecars from extracted camera logs."""

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    args = parser.parse_args()
    for directory in args.directories:
        if not directory.is_dir():
            parser.error(f"Not a directory: {directory}")
    for directory in args.directories:
        removed = 0
        for parent, _, filenames in os.walk(directory, followlinks=False):
            for name in filenames:
                if name.startswith("._"):
                    (Path(parent) / name).unlink()
                    removed += 1
        print(f"Removed {removed} macOS sidecars from {directory}")


if __name__ == "__main__":
    main()
