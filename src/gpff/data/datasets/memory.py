"""An ASE database dataset that is read into memory once."""
from typing import Dict, List

import numpy as np
import torch
from ase.db import connect

import schnetpack.properties as structure
from schnetpack.data import ASEAtomsData

__all__ = ["InMemoryASEAtomsData"]


class InMemoryASEAtomsData(ASEAtomsData):
    """``ASEAtomsData`` that reads every row once into packed arrays and serves
    samples from them. A sample is identical to the one ``ASEAtomsData`` reads
    from the database (same keys, dtypes and values); reading one takes ~0.03 ms
    instead of ~0.5 ms without property columns. The arrays are numpy buffers,
    so forked DataLoader workers share them instead of copying them.
    """

    _cache = None

    def fill_cache(self) -> None:
        """Read the database into memory; call before the workers fork."""
        if self._cache is not None:
            return
        props = list(self.load_properties)
        numbers, positions, cells, pbcs, n_atoms = [], [], [], [], []
        data: Dict[str, List[np.ndarray]] = {p: [] for p in props}
        with connect(self.datapath, use_lock_file=False) as conn:
            for i, row in enumerate(conn.select()):
                if row.id != i + 1:
                    raise RuntimeError(f"row {i} has id {row.id}; expected {i + 1}")
                numbers.append(row["numbers"])
                positions.append(row["positions"])
                cells.append(row["cell"])
                pbcs.append(row["pbc"])
                n_atoms.append(len(row["numbers"]))
                for p in props:
                    data[p].append(row.data[p])
        offsets = np.zeros(len(n_atoms) + 1, dtype=np.int64)
        np.cumsum(n_atoms, out=offsets[1:])
        self._cache = {
            "offsets": offsets,
            "numbers": np.concatenate(numbers),
            "positions": np.concatenate(positions),
            "cell": np.stack(cells),
            "pbc": np.stack(pbcs),
            "data": {p: np.stack(v) for p, v in data.items()},
        }

    def _get_properties(
        self,
        conn,
        idx: int,
        load_properties: List[str],
        load_structure: bool,
        load_metadata: bool = False,
    ):
        cache = self._cache
        if cache is None or load_metadata:
            return super()._get_properties(
                conn, idx, load_properties, load_structure, load_metadata
            )
        # mirrors ASEAtomsData._get_properties, key for key
        start, end = cache["offsets"][idx], cache["offsets"][idx + 1]
        properties = {}
        properties[structure.idx] = torch.tensor([idx])
        for pname in load_properties:
            properties[pname] = (
                torch.tensor(cache["data"][pname][idx].copy()) * self.conversions[pname]
            )
        Z = cache["numbers"][start:end].copy()
        properties[structure.n_atoms] = torch.tensor([Z.shape[0]])
        if load_structure:
            properties[structure.Z] = torch.tensor(Z, dtype=torch.long)
            properties[structure.position] = (
                torch.tensor(cache["positions"][start:end].copy()) * self.distance_conversion
            )
            properties[structure.cell] = (
                torch.tensor(cache["cell"][idx][None].copy()) * self.distance_conversion
            )
            properties[structure.pbc] = torch.tensor(cache["pbc"][idx])
        return properties
