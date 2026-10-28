"""Build a dataset folder from its source files (``gpff_prepare_dataset``).

Each build writes a new folder that gpff trains on with run.data_dir=<folder>
(and samples from with datapath=<folder>/<name>.db split_path=<folder>/split.npz):

    crossdocked   crossdocked_ligand.db    heavy-atom ligands, one row per
                                           (pocket, ligand) pair of the split file
                  crossdocked_complex.db   the same rows, ligand atoms followed
                                           by the pocket atoms
                  crossdocked_complex_pocket_ids.json
                                           pocket atom ids of the test complexes,
                                           for condition_generator=precomputed_scaffold
    geom_drugs    geom_drugs.db            one row per conformer, with hydrogens
                                           (heavy atoms only with --heavy_atoms)

    both          split.npz                train_idx / val_idx / test_idx
                  manifest.json            sources, settings, counts and the
                                           sha256 of every file

The folder is created as <data_root>/<id>, <id> being the first 16 hex digits of
the sha256 over the sha256 of its files, and only once it is complete. Nothing is
downloaded; the sources are read where they are:

    gpff_prepare_dataset crossdocked --root <pocket10 dir> \\
        --split_path <split_by_name.pt> --data_root data/crossdocked
    gpff_prepare_dataset geom_drugs \\
        --source_dir <tensorflow_datasets>/geom/drugs/0.1.4 --data_root data/geom_drugs
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

from gpff.data.datasets.crossdocked import build_crossdocked
from gpff.data.datasets.geom import build_geom_drugs


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    datasets = parser.add_subparsers(dest="dataset", required=True)

    crossdocked = datasets.add_parser(
        "crossdocked", help="CrossDocked2020 pocket10 pairs of the official split."
    )
    crossdocked.add_argument("--root", required=True, type=Path,
                             help="Directory the paths in the split file are relative to.")
    crossdocked.add_argument("--split_path", required=True, type=Path,
                             help="The official split file, split_by_name.pt.")
    crossdocked.add_argument("--n_val", type=int, default=5000,
                             help="Least number of train pairs whose pockets become "
                                  "the validation split. Default: 5000.")
    crossdocked.add_argument("--seed", type=int, default=20261020,
                             help="Seed of the validation pocket draw.")

    geom = datasets.add_parser("geom_drugs", help="GEOM-Drugs from its TFDS export.")
    geom.add_argument("--source_dir", required=True, type=Path,
                      help="Directory of the TFDS export (dataset_info.json and shards).")
    geom.add_argument("--max_conformers", type=int, default=None,
                      help="Keep at most this many conformers per molecule and split.")
    geom.add_argument("--max_shards", type=int, default=None,
                      help="Read only the first shards of each split (trial builds).")
    geom.add_argument("--heavy_atoms", action="store_true",
                      help="Drop the hydrogens of every conformer.")

    for subparser in (crossdocked, geom):
        subparser.add_argument("--data_root", required=True, type=Path,
                               help="Directory the dataset folder is created in.")
        subparser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.data_root.mkdir(parents=True, exist_ok=True)
    partial_dir = args.data_root / f".partial-{args.dataset}-{os.getpid()}"
    partial_dir.mkdir()
    settings = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in ("data_root", "quiet")
    }

    try:
        if args.dataset == "crossdocked":
            counts = build_crossdocked(
                root=args.root,
                split_path=args.split_path,
                target_dir=partial_dir,
                n_val=args.n_val,
                seed=args.seed,
                progress=not args.quiet,
            )
        else:
            counts = build_geom_drugs(
                source_dir=args.source_dir,
                target_dir=partial_dir,
                max_conformers=args.max_conformers,
                max_shards=args.max_shards,
                heavy_atoms=args.heavy_atoms,
                progress=not args.quiet,
            )

        files = {
            path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in sorted(partial_dir.iterdir())
        }
        dataset_id = hashlib.sha256(
            "".join(f"{entry['sha256']}  {name}\n" for name, entry in files.items()).encode()
        ).hexdigest()[:16]
        manifest = {
            "dataset": args.dataset,
            "dataset_id": dataset_id,
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "settings": settings,
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

    print(json.dumps(counts, indent=2))
    print(f"Wrote {target_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
