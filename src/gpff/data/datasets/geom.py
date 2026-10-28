import glob
import json
import os
import re
import struct
from collections import Counter, defaultdict
from typing import Dict, Iterator, Optional, Tuple

import numpy as np
from ase import Atoms
from ase.db import connect
from tqdm import tqdm

__all__ = ["build_geom_drugs", "read_geom_shard"]

GEOM_DRUGS_DB_FILENAME = "geom_drugs.db"

# gpff split -> split of the TFDS export
TFDS_SPLITS = {"train": "train", "val": "val", "test": "test_small"}


def _varint(buffer: memoryview, i: int) -> Tuple[int, int]:
    value = shift = 0
    while True:
        byte = buffer[i]
        i += 1
        value |= (byte & 0x7F) << shift
        shift += 7
        if byte < 0x80:
            return value, i


def _fields(buffer: memoryview) -> Iterator[Tuple[int, object]]:
    """(field number, value) of each field of a serialized protobuf message;
    varints come out as ints, everything else as raw bytes."""
    i = 0
    while i < len(buffer):
        key, i = _varint(buffer, i)
        number, wire_type = key >> 3, key & 7
        if wire_type == 0:
            value, i = _varint(buffer, i)
        elif wire_type == 2:
            length, i = _varint(buffer, i)
            value, i = buffer[i : i + length], i + length
        elif wire_type == 5:
            value, i = buffer[i : i + 4], i + 4
        elif wire_type == 1:
            value, i = buffer[i : i + 8], i + 8
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire_type}.")
        yield number, value


def _example_features(record: bytes, names) -> Dict[str, list]:
    """The values of the named features of a serialized ``tf.train.Example``,
    each a list of the fields of its bytes, float or int64 list."""
    features = {}
    for _, feature_map in _fields(memoryview(record)):  # Example.features
        for _, entry in _fields(feature_map):  # Features.feature, a map entry
            fields = dict(_fields(entry))  # 1: name, 2: Feature
            name = bytes(fields[1]).decode()
            if name in names:
                for _, value_list in _fields(fields[2]):  # the one list it holds
                    features[name] = [value for _, value in _fields(value_list)]
    return features


def _ints(values: list) -> np.ndarray:
    """The values of an int64 list, packed or not."""
    out = []
    for value in values:
        if isinstance(value, int):
            out.append(value)
            continue
        packed = np.frombuffer(value, dtype=np.uint8)
        if (packed < 0x80).all():  # every value fits into one byte
            out.extend(packed.tolist())
        else:
            i = 0
            while i < len(value):
                number, i = _varint(value, i)
                out.append(number)
    return np.array(out, dtype=np.int64)


def _tfrecords(path: str) -> Iterator[bytes]:
    """The records of a TFRecord file (checksums are not verified)."""
    with open(path, "rb") as file:
        while header := file.read(12):
            (length,) = struct.unpack("<Q", header[:8])
            record = file.read(length)
            file.read(4)
            if len(record) != length:
                raise ValueError(f"{path} ends inside a record.")
            yield record


def read_geom_shard(path: str) -> Iterator[Tuple[str, np.ndarray, np.ndarray]]:
    """(SMILES, atomic numbers, positions in Å) of every conformer in a TFRecord
    shard of the GEOM TFDS export."""
    for record in _tfrecords(path):
        features = _example_features(record, ("smiles", "graph_atomic_numbers", "graph_x1"))
        numbers = _ints(features["graph_atomic_numbers"])
        positions = np.frombuffer(
            b"".join(features["graph_x1"]), dtype="<f4"
        ).reshape(-1, 3)
        if len(numbers) != len(positions):
            raise ValueError(f"A record in {path} has {len(numbers)} atoms but {len(positions)} positions.")
        yield bytes(features["smiles"][0]).decode(), numbers, positions


def _split_shards(source_dir: str, tfds_split: str, shard_lengths: list) -> list:
    """The shard files of a split in shard order, checked against the shard count
    of the dataset info."""
    pattern = re.compile(rf"geom-{tfds_split}\.tfrecord-(\d+)-of-(\d+)$")
    shards = {}
    for path in glob.glob(os.path.join(source_dir, f"geom-{tfds_split}.tfrecord-*")):
        match = pattern.search(path)
        if match is None or int(match.group(2)) != len(shard_lengths):
            raise ValueError(f"Unexpected shard {path} for {len(shard_lengths)} shards.")
        shards[int(match.group(1))] = path
    if sorted(shards) != list(range(len(shard_lengths))):
        raise ValueError(
            f"{source_dir} has {len(shards)} of the {len(shard_lengths)} shards of "
            f"the {tfds_split} split."
        )
    return [shards[i] for i in range(len(shard_lengths))]


def build_geom_drugs(
    source_dir: str,
    target_dir: str,
    max_conformers: Optional[int] = None,
    max_shards: Optional[int] = None,
    progress: bool = True,
) -> Dict:
    """Write the conformers of the GEOM-Drugs TFDS export to one database.

    The source is the TFDS export of GEOM-Drugs (``geom/drugs/<version>``: the
    dataset_info.json and the geom-{train,val,test_small}.tfrecord-* shards).
    Its train/val/test_small splits become the train/val/test splits, so the
    database rows are the train conformers, then the val ones, then the test
    ones, in shard and record order. Each row holds one conformer with all its
    atoms (hydrogens included) and its molecule's SMILES as the key-value pair
    "smiles". Only atomic numbers, positions and SMILES are read; the other
    features of the export are ignored.

    Args:
        source_dir: directory of the TFDS export.
        target_dir: existing, empty directory to write the database and the
            split file (split.npz) into.
        max_conformers: keep at most this many conformers per molecule and split,
            the first ones in shard and record order. The export does not carry
            conformer energies, so these are not the lowest-energy ones.
        max_shards: read only the first shards of each split, for trial builds.
        progress: show a progress bar.

    Returns:
        Row, molecule and element counts, and the number of molecules that two
        splits share.
    """
    with open(os.path.join(source_dir, "dataset_info.json")) as file:
        info = json.load(file)
    shard_lengths = {split["name"]: split["shardLengths"] for split in info["splits"]}

    db_path = os.path.join(target_dir, GEOM_DRUGS_DB_FILENAME)
    if os.path.exists(db_path):
        raise FileExistsError(f"There already is a database at {db_path}.")

    partitions, molecules, elements = {}, {}, Counter()
    n_rows = 0
    with connect(db_path, use_lock_file=False) as db:
        for name, tfds_split in TFDS_SPLITS.items():
            shards = _split_shards(source_dir, tfds_split, shard_lengths[tfds_split])
            lengths = [int(n) for n in shard_lengths[tfds_split]]
            if max_shards is not None:
                shards, lengths = shards[:max_shards], lengths[:max_shards]

            first_row = n_rows
            conformers = defaultdict(int)
            with tqdm(total=sum(lengths), desc=f"Reading {name}", disable=not progress) as bar:
                for shard, length in zip(shards, lengths):
                    n_records = 0
                    for smiles, numbers, positions in read_geom_shard(shard):
                        n_records += 1
                        if max_conformers is not None and conformers[smiles] >= max_conformers:
                            continue
                        conformers[smiles] += 1
                        elements.update(numbers.tolist())
                        db.write(Atoms(numbers=numbers, positions=positions), smiles=smiles)
                        n_rows += 1
                    if n_records != length:
                        raise ValueError(f"{shard} has {n_records} records, not {length}.")
                    bar.update(n_records)

            partitions[name] = np.arange(first_row, n_rows, dtype=np.int64)
            molecules[name] = set(conformers)

        db.metadata = {
            "source_dir": str(source_dir),
            "source_version": info.get("version"),
            "source_splits": TFDS_SPLITS,
            "max_conformers": max_conformers,
            "max_shards": max_shards,
            "_property_unit_dict": {},
            "_distance_unit": "Ang",
        }

    np.savez(
        os.path.join(target_dir, "split.npz"),
        **{f"{name}_idx": rows for name, rows in partitions.items()},
    )

    return {
        "n_rows": n_rows,
        **{f"n_{name}": len(rows) for name, rows in partitions.items()},
        **{f"n_{name}_molecules": len(smiles) for name, smiles in molecules.items()},
        "molecules_shared_train_val": len(molecules["train"] & molecules["val"]),
        "molecules_shared_train_test": len(molecules["train"] & molecules["test"]),
        "molecules_shared_val_test": len(molecules["val"] & molecules["test"]),
        "elements": {int(z): n for z, n in sorted(elements.items())},
    }
