"""Report which execution providers Foundry Local can use on this machine.

Inference speed depends almost entirely on which execution provider is active.
Without an accelerated provider the runtime falls back to CPU, which on a small
laptop turns a short answer into minutes rather than seconds.

    python scripts/check_hardware.py            report only
    python scripts/check_hardware.py --install  download and register providers
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import foundry_local_sdk as fl  # noqa: E402

import config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Foundry Local hardware use.")
    parser.add_argument(
        "--install",
        action="store_true",
        help="Download and register the execution providers this device supports.",
    )
    arguments = parser.parse_args()

    fl.FoundryLocalManager.initialize(fl.Configuration(app_name=config.APP_NAME))
    manager = fl.FoundryLocalManager.instance

    providers = manager.discover_eps()
    print(f"Execution providers discovered: {len(providers)}")
    for provider in providers:
        fields = {
            name: getattr(provider, name)
            for name in dir(provider)
            if not name.startswith("_") and not callable(getattr(provider, name))
        }
        print(f"  {fields}")

    if arguments.install:
        print("\nDownloading and registering providers...")
        result = manager.download_and_register_eps(
            progress_callback=lambda name, pct: print(
                f"\r  {name}: {pct:.1f}%", end="", flush=True
            )
        )
        print()
        print(f"Result: {result}")
        for name in dir(result):
            if not name.startswith("_") and not callable(getattr(result, name)):
                print(f"  {name} = {getattr(result, name)}")


if __name__ == "__main__":
    main()
