"""Console entry points for gpff.

One module per command, each registered in ``pyproject.toml``:

    gpff_train              gpff.cli.train:train
    gpff_generate           gpff.cli.generate:sample_molecules
    gpff_prepare_metadata   gpff.cli.prepare_metadata:main
    gpff_calculate_metrics  gpff.cli.calculate_metrics:main
    gpff_diagnose           gpff.cli.diagnose:main
    gpff_reference_assets   gpff.cli.reference_assets:main

This module holds only what every command shares. Cheap imports only
so that running one command does not drag in another's dependencies.
"""
import tempfile
import uuid

import petname
from omegaconf import OmegaConf

from gpff import keys

OmegaConf.register_resolver("uuid", lambda x: str(uuid.uuid1()))
OmegaConf.register_resolver("petname", lambda x: petname.generate(), use_cache=True)
OmegaConf.register_resolver("tmpdir", tempfile.mkdtemp, use_cache=True)
OmegaConf.register_resolver("mul", lambda x, y: x * y)
OmegaConf.register_resolver("key", lambda key_name: getattr(keys, key_name))

header = r""" --- Hello from Generative Pseudo-Force Field (GPFF) model repo! --- \n"""

__all__ = ["header"]
