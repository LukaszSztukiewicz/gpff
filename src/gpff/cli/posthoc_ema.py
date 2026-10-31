"""Reconstruct GPFF inference models: python -m gpff.cli.posthoc_ema."""
import argparse
import json
from pathlib import Path

import torch

from gpff.posthoc_ema import reconstruct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Run's posthoc_ema directory")
    parser.add_argument("--output", type=Path, required=True, help="New inference model path")
    parser.add_argument("--sigma-rel", type=float, required=True)
    parser.add_argument("--step", type=int, help="Saved optimizer-update endpoint (default: latest)")
    args = parser.parse_args()
    metadata_path = Path(str(args.output) + ".json")
    if args.output.exists() or metadata_path.exists():
        parser.error("Output already exists; choose a new path")
    model, metadata = reconstruct(args.input, args.sigma_rel, args.step)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, args.output)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {args.output}; relative profile L2 error: {metadata['relative_profile_error']:.3g}")


if __name__ == "__main__":
    main()
