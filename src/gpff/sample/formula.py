"""Empirical JOINT formula prior fitted solely from explicit training indices."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from schnetpack import properties as P


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def build_formula_prior(dataset_path, split_path, output_path, elements=(1, 6, 7, 8, 9)):
    from ase.db import connect
    with np.load(split_path) as split:
        indices = np.asarray(split["train_idx"], dtype=np.int64)
        others = np.concatenate([split["val_idx"], split["test_idx"]])
    if len(set(indices)) != len(indices) or np.intersect1d(indices, others).size:
        raise ValueError("Training indices must be unique and disjoint from held-out data")
    counts = Counter()
    with connect(dataset_path) as db:
        for index in indices:
            z = db.get(int(index) + 1).numbers  # split is zero-based, ASE IDs start at 1
            if not np.isin(z, elements).all():
                raise ValueError("Training formula contains unsupported elements")
            counts[tuple(int((z == k).sum()) for k in elements)] += 1
    payload = {"elements": list(elements), "formulas": [list(c) for c in counts],
               "frequencies": list(counts.values()), "training_samples": len(indices),
               "train_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
               "dataset_sha256": file_sha256(dataset_path), "split_sha256": file_sha256(split_path)}
    with Path(output_path).open("x") as stream:
        json.dump(payload, stream, indent=2)
    return payload


class FormulaPrior:
    def __init__(self, formula_path, sigma=30.0, mode="conditional_size", seed=0):
        if mode not in ("conditional_size", "joint"):
            raise ValueError("Unknown formula mode")
        self.payload = json.loads(Path(formula_path).read_text())
        self.formulas = torch.tensor(self.payload["formulas"], dtype=torch.long)
        self.frequencies = torch.tensor(self.payload["frequencies"], dtype=torch.float64)
        if self.formulas.ndim != 2 or len(self.frequencies) != len(self.formulas) or not len(self.formulas):
            raise ValueError("Invalid formula table")
        if bool((self.formulas < 0).any()) or bool((self.frequencies <= 0).any()) or bool((self.formulas.sum(1) == 0).any()):
            raise ValueError("Invalid formula counts/frequencies")
        self.elements = torch.tensor(self.payload["elements"], dtype=torch.long)
        self.sigma, self.mode = sigma, mode
        # Independent formula/assignment and position streams for paired runs.
        self.rng = torch.Generator().manual_seed(seed)
        self.position_rng = torch.Generator().manual_seed(seed + 1)
        self.requested = []

    def sample_from_batch(self, batch, conditions=None):
        sizes = batch[P.n_atoms].cpu().tolist()
        formulas, zs, positions = [], [], []
        for size in sizes:
            weights = self.frequencies.clone()
            if self.mode == "conditional_size":
                weights *= self.formulas.sum(1) == size
            if not bool(weights.sum() > 0):
                raise ValueError(f"No training formula for size {size}")
            index = torch.multinomial(weights, 1, generator=self.rng).item()
            formula = self.formulas[index]
            z = torch.repeat_interleave(self.elements, formula)
            z = z[torch.randperm(len(z), generator=self.rng)]
            zs.append(z)
            positions.append(torch.randn((len(z), 3), generator=self.position_rng) * self.sigma)
            formulas.append(formula.tolist())
        ns = torch.tensor([len(z) for z in zs], dtype=torch.long)
        first_index = len(self.requested)
        self.requested.extend(formulas)
        # Fresh batch: no test coordinates, elements, neighbor lists or properties.
        return {P.Z: torch.cat(zs), P.R: torch.cat(positions), P.n_atoms: ns,
                P.idx: torch.arange(first_index,first_index+len(ns),dtype=torch.long),
                P.idx_m: torch.arange(len(ns)).repeat_interleave(ns),
                P.cell: torch.zeros((len(ns), 3, 3)), P.pbc: torch.zeros((len(ns), 3), dtype=torch.bool),
                "requested_formula": torch.tensor(formulas, dtype=torch.long)}

    def verify(self, batch):
        requested = batch["requested_formula"].to(batch[P.Z].device)
        actual = torch.zeros_like(requested)
        for i, z in enumerate(self.elements.tolist()):
            actual[:, i].index_add_(0, batch[P.idx_m], (batch[P.Z] == z).long())
        if not torch.equal(actual, requested) or int(actual.sum()) != len(batch[P.Z]):
            raise ValueError("Generator did not preserve the requested formula")
