"""Console entry points for gpff.

One module per command, each registered in ``pyproject.toml``:

    gpff_train              gpff.cli.train:train
    gpff_generate           gpff.cli.generate:sample_molecules
    gpff_prepare_metadata   gpff.cli.prepare_metadata:main
    gpff_calculate_metrics  gpff.cli.calculate_metrics:main
    gpff_reference_assets   gpff.cli.reference_assets:main

This module holds only what every command shares: the banner, the
``torch.load`` compatibility shim and the OmegaConf resolvers. Keep it free of
heavy imports so that running one command does not drag in another's
dependencies.
"""
import tempfile
import uuid

import petname
import torch
from omegaconf import OmegaConf

from gpff import keys

# torch >= 2.6 defaults torch.load(weights_only=True), which cannot unpickle the
# full Lightning checkpoints written by this trainer (they contain schnetpack
# model objects). pytorch-lightning passes weights_only=True explicitly, so we
# force it back to False. Safe because we only ever load our own checkpoints.
_orig_torch_load = torch.load


def _torch_load_compat(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_compat


OmegaConf.register_new_resolver("uuid", lambda x: str(uuid.uuid1()))
OmegaConf.register_new_resolver("petname", lambda x: petname.generate(), use_cache=True)
OmegaConf.register_new_resolver("tmpdir", tempfile.mkdtemp, use_cache=True)
OmegaConf.register_new_resolver("mul", lambda x, y: x * y)
OmegaConf.register_new_resolver("key", lambda key_name: getattr(keys, key_name))

header = r"""
 ___________ _   __           _____           ___  ___ _     ____________ 
/  ___| ___ \ | / /          |  __ \          |  \/  || |    |  ___|  ___|
\ `--.| |_/ / |/ /   ______  | |  \/ ___ _ __ | .  . || |    | |_  | |_   
 `--. \  __/|    \  |______| | | __ / _ \ '_ \| |\/| || |    |  _| |  _|  
/\__/ / |   | |\  \          | |_\ \  __/ | | | |  | || |____| |   | |    
\____/\_|   \_| \_/           \____/\___|_| |_\_|  |_/\_____/\_|   \_|    

"""

__all__ = ["header"]
