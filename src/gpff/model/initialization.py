"""Audited weights-only restart, with an exact allowlist for new sigma modules."""
import json
from pathlib import Path
from schnetpack.utils import load_model
from gpff.sample.formula import file_sha256


def initialize_weights(model, path, allow_sigma=False, report_path="initialization.json"):
    source = load_model(path, device="cpu")
    incoming, target = source.state_dict(), model.state_dict()
    missing, extra = set(target) - set(incoming), set(incoming) - set(target)
    allowed = {"representation.sigma_embedding." + suffix for suffix in
               ("0.weight", "0.bias", "2.weight", "2.bias")} if allow_sigma else set()
    if missing != allowed or extra:
        raise ValueError(f"Unexpected initialization keys: missing={sorted(missing)}, extra={sorted(extra)}, allowed={sorted(allowed)}")
    # Check shapes and all keys strictly, including initialized new parameters.
    merged = dict(target)
    merged.update(incoming)
    model.load_state_dict(merged, strict=True)
    payload = {"source": str(path), "source_sha256": file_sha256(path),
               "new_parameters": sorted(missing), "optimizer": "fresh", "scheduler": "fresh"}
    Path(report_path).write_text(json.dumps(payload, indent=2) + "\n")
    return payload
