import logging
import os
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Optional

import numpy as np
from schnetpack.data import (
    AtomsDataFormat,
    AtomsDataModule,
    AtomsDataModuleError,
    AtomsLoader,
    load_dataset,
)

from gpff.data.datasets.memory import InMemoryASEAtomsData

__all__ = ["PreparedAtomsData", "database_n_atoms"]

log = logging.getLogger(__name__)


def database_n_atoms(datapath) -> np.ndarray:
    """The number of atoms of every row of an ASE database, by dataset index.

    Read in one statement from the ``natoms`` column that ASE keeps for its queries,
    instead of from the rows themselves. A dataset index is the row id minus one
    (``ASEAtomsData`` reads index i from row i + 1), so the ids must be 1..n.
    """
    uri = Path(datapath).resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        rows = connection.execute("SELECT id, natoms FROM systems ORDER BY id").fetchall()
    ids, n_atoms = np.array(rows, dtype=np.int64).reshape(-1, 2).T
    if not np.array_equal(ids, np.arange(1, len(ids) + 1)):
        raise AtomsDataModuleError(f"The row ids of {datapath} are not 1..{len(ids)}.")
    return n_atoms


class PreparedAtomsData(AtomsDataModule):
    """
    Data module for an ASE database that already exists, e.g. one written by
    ``gpff_prepare_geom_revisited``. It never downloads or builds a database: a missing
    ``datapath`` is an error. It can keep only the molecules with at most a number of
    atoms.
    """

    def __init__(
        self,
        datapath: str,
        batch_size: int,
        max_n_atoms: Optional[int] = None,
        cache_in_memory: bool = False,
        persistent_workers: bool = False,
        format: Optional[AtomsDataFormat] = AtomsDataFormat.ASE,
        **kwargs
    ):
        """
        Args:
            datapath: path to the ASE database.
            batch_size: batch size.
            max_n_atoms: keep only the molecules with at most this many atoms, in
                          every split (e.g. a size curriculum); the split file
                          is filtered, not replaced.
            cache_in_memory: read the database into memory once in setup and
                          serve samples from there instead of from the database.
            persistent_workers: keep the DataLoader workers alive between epochs.
            format: format of the database.
        """
        super().__init__(datapath=datapath, batch_size=batch_size, format=format, **kwargs)
        self.max_n_atoms = max_n_atoms
        self.cache_in_memory = cache_in_memory
        self.persistent_workers = persistent_workers

    def prepare_data(self):
        if not os.path.exists(self.datapath):
            raise AtomsDataModuleError(
                f"There is no database at {self.datapath}; build it first, "
                "e.g. with gpff_prepare_geom_revisited."
            )

    def setup(self, stage: Optional[str] = None):
        """
        Overwrites the ``setup`` method to optionally serve the database from
        memory and to keep only the molecules with at most ``max_n_atoms`` atoms.
        """
        if self.data_workdir is None:
            datapath = self.datapath
        else:
            datapath = self._copy_to_workdir()

        # (re)load datasets
        if self.dataset is None:
            if self.cache_in_memory:
                self.dataset = InMemoryASEAtomsData(
                    datapath,
                    property_units=self.property_units,
                    distance_unit=self.distance_unit,
                    load_properties=self.load_properties,
                )
                self.dataset.fill_cache()
            else:
                self.dataset = load_dataset(
                    datapath,
                    self.format,
                    property_units=self.property_units,
                    distance_unit=self.distance_unit,
                    load_properties=self.load_properties,
                )

            # load and generate partitions if needed
            if self.train_idx is None:
                self._load_partitions()

            # use only molecules with at most max_n_atoms atoms, in every split
            if self.max_n_atoms is not None:
                n_atoms = database_n_atoms(datapath)

                def small(idx):
                    idx = np.asarray(idx, dtype=np.int64)
                    return idx[n_atoms[idx] <= self.max_n_atoms].tolist()

                self.train_idx, self.val_idx, self.test_idx = map(
                    small, (self.train_idx, self.val_idx, self.test_idx)
                )
                log.info(
                    f"Keeping the molecules with at most {self.max_n_atoms} atoms: "
                    f"{len(self.train_idx)} train, {len(self.val_idx)} val, "
                    f"{len(self.test_idx)} test"
                )

            # partition dataset
            self._train_dataset = self.dataset.subset(self.train_idx)
            self._val_dataset = self.dataset.subset(self.val_idx)
            self._test_dataset = self.dataset.subset(self.test_idx)

        self._setup_transforms()

    # schnetpack's loaders take no persistent_workers; these mirror them with it
    def train_dataloader(self) -> AtomsLoader:
        if self._train_dataloader is None:
            train_batch_sampler = self._setup_sampler(
                sampler_cls=self.train_sampler_cls,
                sampler_args=self.train_sampler_args,
                dataset=self._train_dataset,
            )
            self._train_dataloader = AtomsLoader(
                self.train_dataset,
                batch_size=self.batch_size if train_batch_sampler is None else 1,
                shuffle=train_batch_sampler is None,
                batch_sampler=train_batch_sampler,
                num_workers=self.num_workers,
                pin_memory=self._pin_memory,
                persistent_workers=self.persistent_workers and self.num_workers > 0,
            )
        return self._train_dataloader

    def val_dataloader(self) -> AtomsLoader:
        if self._val_dataloader is None:
            self._val_dataloader = AtomsLoader(
                self.val_dataset,
                batch_size=self.val_batch_size,
                num_workers=self.num_val_workers,
                pin_memory=self._pin_memory,
                persistent_workers=self.persistent_workers and self.num_val_workers > 0,
            )
        return self._val_dataloader

    def test_dataloader(self) -> AtomsLoader:
        if self._test_dataloader is None:
            self._test_dataloader = AtomsLoader(
                self.test_dataset,
                batch_size=self.test_batch_size,
                num_workers=self.num_test_workers,
                pin_memory=self._pin_memory,
                persistent_workers=self.persistent_workers and self.num_test_workers > 0,
            )
        return self._test_dataloader
