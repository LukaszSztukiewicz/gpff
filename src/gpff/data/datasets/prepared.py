import os
import pickle
from typing import Optional

from tqdm import tqdm

from schnetpack import properties
from schnetpack.data import (
    AtomsDataFormat,
    AtomsDataModule,
    AtomsDataModuleError,
    AtomsLoader,
    load_dataset,
)

from gpff.data.datasets.memory import InMemoryASEAtomsData

__all__ = ["PreparedAtomsData"]


class PreparedAtomsData(AtomsDataModule):
    """
    Data module for an ASE database that already exists, e.g. one written by
    ``gpff_prepare_dataset``. It never downloads or builds a database: a missing
    ``datapath`` is an error. Also allows to only load molecules with a specified
    number of atoms.
    """

    def __init__(
        self,
        datapath: str,
        batch_size: int,
        n_atoms_allowed: Optional[int] = None,
        indices_path: str = "n_atoms_indices.pkl",
        cache_in_memory: bool = False,
        persistent_workers: bool = False,
        format: Optional[AtomsDataFormat] = AtomsDataFormat.ASE,
        **kwargs
    ):
        """
        Args:
            datapath: path to the ASE database.
            batch_size: batch size.
            n_atoms_allowed: the exact number of atoms in each molecule.
            indices_path: path to pickle file containing indices of molecules
                          with specific number of atoms.
            cache_in_memory: read the database into memory once in setup and
                          serve samples from there instead of from the database.
            persistent_workers: keep the DataLoader workers alive between epochs.
            format: format of the database.
        """
        super().__init__(datapath=datapath, batch_size=batch_size, format=format, **kwargs)
        self.n_atoms_allowed = n_atoms_allowed
        self.indices_path = indices_path
        self.cache_in_memory = cache_in_memory
        self.persistent_workers = persistent_workers

    def prepare_data(self):
        if not os.path.exists(self.datapath):
            raise AtomsDataModuleError(
                f"There is no database at {self.datapath}; build it first, "
                "e.g. with gpff_prepare_dataset."
            )

    def setup(self, stage: Optional[str] = None):
        """
        Overwrites the ``setup`` method to load molecules with given number of atoms.
        Copied from MoreRed.
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

            # use only molecules with specific number of atoms
            if self.n_atoms_allowed is not None and self.n_atoms_allowed > 0:
                # load indices
                if os.path.exists(self.indices_path):
                    with open(self.indices_path, "rb") as file:
                        indices = pickle.load(file)
                else:
                    indices = {}

                # get indices of molecules with specific number of atoms
                if self.n_atoms_allowed in indices.keys():
                    indices = indices[self.n_atoms_allowed]
                else:
                    tmp = []
                    for i in tqdm(range(len(self.dataset)), f"getting molecules with {self.n_atoms_allowed} atoms"):
                        if self.dataset[i][properties.n_atoms] == self.n_atoms_allowed:  # type: ignore
                            tmp.append(i)
                    indices[self.n_atoms_allowed] = tmp

                    # persist indices
                    with open(self.indices_path, "wb") as file:
                        pickle.dump(indices, file)
                    indices = indices[self.n_atoms_allowed]

            # get all indices (with any number of atoms)
            else:
                indices = list(range(len(self.dataset)))

            # subset dataset
            self.dataset = self.dataset.subset(indices)

            # load and generate partitions if needed
            if self.train_idx is None:
                self._load_partitions()

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
