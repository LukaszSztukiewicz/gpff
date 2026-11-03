"""Build the GEOM-Drugs dataset of the MiDi split, cleaned as in GEOM-Drugs Revisited
(``gpff_prepare_geom_revisited``).

The build writes a new folder <data_root>/<id>, <id> being the first 16 hex digits
of the sha256 over the sha256 of its files, and only once it is complete:

    geom_drugs.db     one cleaned conformer per row, hydrogens included
    split_all.npz     train_idx / val_idx / test_idx: every conformer
    split_top5.npz    the same, only the first 5 conformers of each molecule
    manifest.json     source, cleaning, settings, counts and the sha256 of every file

A run picks the conformers it trains on by its data config, with run.data_dir=<folder>:

    gpff_train data=geom_drugs_revisited_all  run.data_dir=<folder> ...
    gpff_train data=geom_drugs_revisited_top5 run.data_dir=<folder> ...

Nothing is downloaded; the pickles of https://doi.org/10.5281/zenodo.19150583 are
read where they are, and refused unless they have the MD5 of that record:

    gpff_prepare_geom_revisited --source_dir <folder of the 3 pickles> \\
        --data_root data/geom_drugs --n_workers 8

See gpff.data.datasets.geom_revisited for what is done to the molecules.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional

import rdkit

from gpff.data.datasets.geom_revisited import (
    UPSTREAM,
    ZENODO_DOI,
    ZENODO_MD5,
    build_geom_drugs_revisited,
)

PROTOCOL = "geom_drugs_revisited"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source_dir", required=True, type=Path,
                        help="Folder of train_data.pickle, val_data.pickle and test_data.pickle.")
    parser.add_argument("--data_root", required=True, type=Path,
                        help="Directory the dataset folder is created in.")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Conformers per molecule of the second split file. Default: 5.")
    parser.add_argument("--max_molecules", type=int, default=None,
                        help="Read only the first molecules of each split (trial builds; "
                             "use a data_root of their own).")
    parser.add_argument("--n_workers", type=int, default=1,
                        help="Processes that clean molecules. The build is the same for any number.")
    parser.add_argument("--allow_unknown_source", action="store_true",
                        help="Unpickle files whose MD5 is not the one of the Zenodo record.")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    args.data_root.mkdir(parents=True, exist_ok=True)
    partial_dir = args.data_root / f".partial-{PROTOCOL}-{os.getpid()}"
    partial_dir.mkdir()
    settings = {
        "dataset": "geom_drugs",
        "protocol": PROTOCOL,
        "source_dir": str(args.source_dir),
        "top_k": args.top_k,
        "max_molecules": args.max_molecules,
        "n_workers": args.n_workers,
        "heavy_atoms": False,
        "allow_unknown_source": args.allow_unknown_source,
    }

    try:
        counts = build_geom_drugs_revisited(
            source_dir=args.source_dir,
            target_dir=partial_dir,
            top_k=args.top_k,
            max_molecules=args.max_molecules,
            n_workers=args.n_workers,
            known_md5=None if args.allow_unknown_source else ZENODO_MD5,
            progress=not args.quiet,
        )
        sources = counts.pop("sources")
        statistics = counts.pop("cleaning")

        files = {
            path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in sorted(partial_dir.iterdir())
        }
        dataset_id = hashlib.sha256(
            "".join(f"{entry['sha256']}  {name}\n" for name, entry in files.items()).encode()
        ).hexdigest()[:16]
        manifest = {
            "dataset": "geom_drugs",
            "protocol": PROTOCOL,
            "dataset_id": dataset_id,
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "settings": settings,
            "source": {"doi": ZENODO_DOI, "files": sources},
            "cleaning": {
                "script": UPSTREAM,
                "rdkit": rdkit.__version__,
                "rule": "per molecule: the SMILES parses; conformers are dropped if they fail "
                        "sanitization, kekulization or the single-fragment check; the molecule is "
                        "removed if none is left or any left has a bond that is not within 40% of "
                        "the sum of the covalent radii",
                "statistics": statistics,
            },
            "counts": counts,
            "files": files,
        }
        (partial_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

        target_dir = args.data_root / dataset_id
        if target_dir.exists():
            raise FileExistsError(f"There already is a dataset at {target_dir}.")
        partial_dir.rename(target_dir)
    finally:
        if partial_dir.exists():
            shutil.rmtree(partial_dir)

    print(json.dumps({"counts": counts, "cleaning": statistics}, indent=2))
    print(f"Wrote {target_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
