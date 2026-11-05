"""Console entry points for gpff.

One module per command, each registered in ``pyproject.toml``:

    gpff_train              gpff.cli.train:train
    gpff_generate           gpff.cli.generate:sample_molecules
    gpff_prepare_metadata   gpff.cli.prepare_metadata:main
    gpff_calculate_metrics  gpff.cli.calculate_metrics:main
    gpff_reference_assets   gpff.cli.reference_assets:main
    gpff_report             gpff.cli.report:main

This module holds only what every command shares. Cheap imports only
so that running one command does not drag in another's dependencies.
"""
import hashlib
import json
import tempfile
import time
import uuid
from importlib import metadata
from pathlib import Path

import petname
from omegaconf import OmegaConf

from gpff import keys

OmegaConf.register_resolver("uuid", lambda x: str(uuid.uuid1()))
OmegaConf.register_resolver("petname", lambda x: petname.generate(), use_cache=True)
OmegaConf.register_resolver("tmpdir", tempfile.mkdtemp, use_cache=True)
OmegaConf.register_resolver("mul", lambda x, y: x * y)
OmegaConf.register_resolver("key", lambda key_name: getattr(keys, key_name))

header = r""" --- Hello from Generative Pseudo-Force Field (GPFF) model repo! --- \n"""


def _md5(path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _version(package: str):
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def record_provenance(evaluation_dir, section: str, names, inputs=None) -> None:
    """Note which code computed each of `names` in <evaluation_dir>/provenance.json.

    The file keeps one entry per array ("metadata" section) or metric ("results"
    section), replaced whenever it is computed again, so a results.json that
    several code versions wrote into can still be told apart name by name. An
    entry holds the path of the gpff package that ran (on the cluster, the
    frozen_code snapshot), the RDKit and PoseBusters versions, the path and md5
    of every input file given (`inputs`: argument name -> path or None), and the
    time.
    """
    path = Path(evaluation_dir) / "provenance.json"
    provenance = json.loads(path.read_text()) if path.exists() else {}
    entry = {
        "code": str(Path(__file__).resolve().parents[1]),
        "rdkit": _version("rdkit"),
        "posebusters": _version("posebusters"),
        "inputs": {
            name: {"path": str(input_path), "md5": _md5(input_path)}
            for name, input_path in (inputs or {}).items()
            if input_path is not None
        },
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    provenance.setdefault(section, {}).update({name: entry for name in names})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")


__all__ = ["header", "record_provenance"]
