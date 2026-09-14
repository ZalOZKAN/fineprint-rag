"""Inspect which build of a model Foundry Local is actually running.

Registering the execution providers made generation 5.5 times faster, but
nvidia-smi reported 0 percent GPU utilisation throughout an evaluation run, so
the NVIDIA GPU is idle and the speedup came from somewhere else.

A model in the catalog ships as several variants, each compiled for a particular
execution provider. If only a CPU variant exists locally, registering CUDA
changes nothing for that model. This script prints every variant and which one is
selected, which is the difference between "CUDA is unavailable" and "CUDA is
available and unused".

    python scripts/inspect_variants.py
    python scripts/inspect_variants.py qwen3-4b
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import foundry_local_sdk as fl  # noqa: E402

import config  # noqa: E402


def fields(obj: object) -> dict:
    """Readable, non callable attributes of an object."""
    result = {}
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            value = getattr(obj, name)
        except Exception:  # noqa: BLE001 - deprecated attributes raise
            continue
        if not callable(value):
            result[name] = value
    return result


def main() -> None:
    alias = sys.argv[1] if len(sys.argv) > 1 else config.CHAT_MODEL

    fl.FoundryLocalManager.initialize(fl.Configuration(app_name=config.APP_NAME))
    manager = fl.FoundryLocalManager.instance

    print("registered execution providers:")
    for provider in manager.discover_eps():
        print(f"  {provider.name}: registered={provider.is_registered}")

    model = manager.catalog.get_model(alias)
    if model is None:
        raise SystemExit(f"Model not in catalog: {alias}")

    print(f"\nmodel: {alias}")
    print(f"  id        {model.id}")
    print(f"  cached    {model.is_cached}")
    print(f"  context   {model.context_length}")

    variants = model.variants
    print(f"\nvariants: {len(variants)}")
    for variant in variants:
        info = fields(variant)
        # Print the fields that identify the hardware target, then the rest.
        interesting = {
            key: value
            for key, value in info.items()
            if any(word in key.lower() for word in ("id", "alias", "device", "ep",
                                                    "provider", "runtime", "cached",
                                                    "execution"))
        }
        print(f"  {interesting}")

    print("\ncached models on this machine:")
    for cached in manager.catalog.get_cached_models():
        print(f"  {cached.id}")


if __name__ == "__main__":
    main()
