from . import keys

# Compatibility alias: the bundled trained checkpoints were pickled under the
# package's previous name ("matred"). Register import aliases so torch.load can
# resolve those qualified class paths against the current "gpff" package.
import sys as _sys
import importlib as _importlib

_MATRED_SUBMODULES = (
    "",
    ".callbacks",
    ".cli",
    ".data",
    ".data.datasets",
    ".data.datasets.qm9",
    ".data.transform",
    ".data.transform.pseudo_pes",
    ".data.transform.scaling",
    ".keys",
    ".model",
    ".model.heads",
    ".model.painn",
    ".nn",
    ".paper",
    ".paper.plotting",
    ".pseudo_pes",
    ".pseudo_pes.geometries",
    ".pseudo_pes.labels",
    ".pseudo_pes.weights",
    ".sample",
    ".sample.priors",
    ".sample.pseudo_ff",
    ".sample.schedules",
    ".task",
    ".utils",
    ".utils.wandb",
)
for _suffix in _MATRED_SUBMODULES:
    try:
        _module = _importlib.import_module(f"gpff{_suffix}")
    except ImportError:
        continue
    _sys.modules.setdefault(f"matred{_suffix}", _module)

del _sys, _importlib, _MATRED_SUBMODULES, _suffix, _module

__all__ = ["keys"]
