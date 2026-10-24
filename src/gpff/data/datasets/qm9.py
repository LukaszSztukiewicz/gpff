import os
from typing import List, Optional

from ase.db import connect
import numpy as np
from tqdm import tqdm

from schnetpack.datasets import QM9
from schnetpack.data import AtomsDataModuleError, AtomsLoader, load_dataset

from gpff.analysis import analyze_with_rdkit, get_rdkit_validity
from gpff.data.datasets.memory import InMemoryASEAtomsData

__all__ = ["QM9Valid", "build_filtered_qm9"]

# name of the raw (unfiltered) QM9 database that is downloaded next to the
# filtered one when the filtered database does not exist yet
RAW_QM9_FILENAME = "qm9.db"

# number of molecules of the filtered databases that were used in the paper
# (the first one predates the switch to `remove_uncharacterized=True`)
KNOWN_FILTERED_SIZES = [121651, 123569]


def build_filtered_qm9(
    raw_datapath: str,
    target_datapath: str,
    timeout: int = 5,
    progress: bool = True,
) -> int:
    """Write all chemically valid molecules of a raw QM9 database to a new database.

    A molecule is kept if ``gpff.analysis.get_rdkit_validity`` accepts it, i.e. if
    RDKit can infer a neutral, radical-free, single-fragment bond structure from
    its geometry. The written database carries the same properties as the source
    plus the index of each molecule in the source database ("original_index")
    and the SMILES string as a key-value pair.

    Args:
        raw_datapath: path to the unfiltered QM9 database.
        target_datapath: path of the filtered database to write.
        timeout: seconds allowed per molecule for the RDKit bond perception.
        progress: show progress bars.

    Returns:
        The number of molecules written.
    """
    if os.path.exists(target_datapath):
        raise FileExistsError(f"There already is a database at {target_datapath}.")

    # find the valid molecules
    with connect(raw_datapath) as source:
        n_molecules = source.count()
        valid_indices = [
            index
            for index, row in enumerate(
                tqdm(
                    source.select(),
                    total=n_molecules,
                    desc="Checking validity",
                    disable=not progress,
                )
            )
            if get_rdkit_validity(row.toatoms(), timeout=timeout)
        ]

    # write to a temporary file first, so that an interrupted run does not leave
    # behind a partial database that looks ready to use
    partial_datapath = os.path.splitext(target_datapath)[0] + "_partial.db"
    if os.path.exists(partial_datapath):
        os.remove(partial_datapath)

    with connect(partial_datapath) as target, connect(raw_datapath) as source:
        source_metadata = source.metadata
        property_unit_dict = dict(source_metadata["_property_unit_dict"])
        property_unit_dict["original_index"] = ""

        for index in tqdm(
            valid_indices,
            desc="Writing valid molecules",
            disable=not progress,
        ):
            row = source.get(index + 1)
            atoms = row.toatoms()
            analysis = analyze_with_rdkit(atoms, timeout=timeout)

            data = dict(row.data)
            data["original_index"] = np.array([index], dtype=int)
            # the rows carry the SMILES without the explicit hydrogens
            target.write(
                atoms, data=data, smiles=analysis.get("canonical_smiles", "")
            )

        target.metadata = {
            "original_data_base_path": str(raw_datapath),
            "filter_flags": ["valid"],
            "settings": {
                "allowed_charges": [0],
                "allow_charged_fragments": False,
                "allow_radical_electrons": False,
                "ignore_enantiomers": False,
                "ignore_isomerism": False,
                "timeout": timeout,
            },
            "_property_unit_dict": property_unit_dict,
            "_distance_unit": source_metadata["_distance_unit"],
        }

    os.replace(partial_datapath, target_datapath)
    return len(valid_indices)


class QM9Valid(QM9):
    """
    Filtered version of QM9 dataset without invalid molecules.
    todo: also allow with uncharacterized molecules
    """

    def __init__(
        self,
        datapath: str,
        batch_size: int,
        load_properties: Optional[List[str]] = None,
        raw_datapath: Optional[str] = None,
        filter_timeout: int = 5,
        cache_in_memory: bool = False,
        persistent_workers: bool = False,
        **kwargs
    ):
        """
        Args:
            datapath: path to directory containing QM9 database.
            batch_size: batch size.
            raw_datapath: path of the unfiltered QM9 database that the filtered
                          database is built from. Defaults to "qm9.db" next to
                          ``datapath``. It is downloaded if it does not exist.
            filter_timeout: seconds allowed per molecule when checking validity.
            cache_in_memory: read the database into memory once in setup and
                          serve samples from there instead of from the database.
            persistent_workers: keep the DataLoader workers alive between epochs.
        """
        super().__init__(
            datapath=datapath,
            batch_size=batch_size,
            load_properties=load_properties,
            **kwargs,
        )
        self.raw_datapath = raw_datapath or os.path.join(
            os.path.dirname(os.path.abspath(datapath)), RAW_QM9_FILENAME
        )
        self.filter_timeout = filter_timeout
        self.cache_in_memory = cache_in_memory
        self.persistent_workers = persistent_workers

    def prepare_data(self):
        if not os.path.exists(self.datapath):
            # download the raw data (without the uncharacterized molecules) and
            # remove the molecules that RDKit does not accept
            QM9(
                datapath=self.raw_datapath,
                batch_size=self.batch_size,
                remove_uncharacterized=True,
            ).prepare_data()
            build_filtered_qm9(
                raw_datapath=self.raw_datapath,
                target_datapath=self.datapath,
                timeout=self.filter_timeout,
            )

        dataset = load_dataset(self.datapath, self.format)

        # check that the invalid molecules have been removed
        filtered = "valid" in dataset.metadata.get("filter_flags", [])
        if not filtered and len(dataset) not in KNOWN_FILTERED_SIZES:
            raise AtomsDataModuleError(
                "The dataset at the chosen location contains the invalid molecules! "
            )

    def setup(self, stage: Optional[str] = None):
        """
        Overwrites the ``setup`` method to optionally serve the database from
        memory and to reset the dataset transforms on every call.
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
