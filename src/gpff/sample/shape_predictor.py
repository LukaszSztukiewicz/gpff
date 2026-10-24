"""Shape predictors supplying per-molecule principal-axis variances to the priors.

A shape predictor answers "how large and how elongated should a molecule of
this many atoms be?". Priors consume its output; see
:class:`gpff.sample.priors.ShapePrior`.
"""
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from schnetpack import properties
from schnetpack.data.atoms import ASEAtomsData
from tqdm import tqdm

from gpff.utils import batch_compute_principal_variances

__all__ = [
    "ShapeGMM",
    "normalize_target_ratio",
    "apply_target_ratio",
]


def normalize_target_ratio(target_ratio: Sequence[float]) -> torch.Tensor:
    """
    Normalize an (xx, yy, zz) principal-axis variance ratio so that it sums to one.
    """
    ratio = torch.as_tensor(target_ratio, dtype=torch.float64).flatten()
    if ratio.shape != (3,):
        raise ValueError(f"target_ratio must have three entries, got {tuple(ratio.shape)}.")
    ratio = ratio.clamp(min=0.0)
    total = ratio.sum()
    if total <= 0.0:
        raise ValueError("target_ratio must contain at least one positive entry.")
    return ratio / total


def apply_target_ratio(variances: torch.Tensor, target_ratio: torch.Tensor) -> torch.Tensor:
    """
    Redistribute the total variance over the principal axes.

    Keeping the sum preserves the molecule size the shape predictor predicted
    while forcing the requested elongation.

    Args:
        variances: Principal-axis variance tensor of shape (3,) or (n_samples, 3).
        target_ratio: Normalized (3,) ratio, see :func:`normalize_target_ratio`.

    Returns:
        Variances of the same shape as the input.
    """
    ratio = target_ratio.to(dtype=variances.dtype, device=variances.device)
    total_variance = variances.sum(dim=-1, keepdim=True)
    return total_variance * ratio


class ShapeGMM:
    """
    Per-molecule-size Gaussian mixture over principal-axis variances.

    One GMM is fitted per atom count, so a sample reflects how large and how
    elongated a molecule of that size typically is. Priors consume the sampled
    variances; the model itself knows nothing about them.
    """

    def __init__(
        self,
        training_set: ASEAtomsData,
        n_components: int = 5,
        variance_floor: float = 1e-6,
        verbose: bool = False,
    ):
        """
        Args:
            training_set: Structures to fit the per-size mixtures on.
            n_components: Number of mixture components per atom count.
            variance_floor: Lower bound applied to every principal variance
                before the log. Planar and linear molecules have a vanishing
                smallest variance, which the log would send to -inf.
            verbose: Show progress bars while fitting.
        """
        self.n_components = n_components
        self.variance_floor = variance_floor
        self.verbose = verbose
        self.models: Dict[int, Any] = {}
        self.fit(training_set)

    @classmethod
    def load_or_fit(
        cls,
        train_dataset: Optional[ASEAtomsData] = None,
        cache_path: Optional[str | Path] = None,
        n_components: int = 5,
        variance_floor: float = 1e-6,
        verbose: bool = False,
    ) -> "ShapeGMM":
        """
        Load a cached shape predictor or fit and cache a new one.
        """
        if cache_path is not None:
            cache_path = Path(cache_path)
            if cache_path.exists():
                return torch.load(cache_path, map_location="cpu", weights_only=False)

        if train_dataset is None:
            raise ValueError("ShapeGMM.load_or_fit needs a train_dataset to fit a missing model.")

        shape_predictor = cls(
            training_set=train_dataset,
            n_components=n_components,
            variance_floor=variance_floor,
            verbose=verbose,
        )
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(shape_predictor, cache_path)
        return shape_predictor

    def fit(self, training_set: ASEAtomsData) -> None:
        from sklearn.mixture import GaussianMixture

        structures: List[torch.Tensor] = []
        sample_iterator = tqdm(
            range(len(training_set)),
            desc="Computing variance training set",
            disable=not self.verbose,
        )
        for sample_idx in sample_iterator:
            structures.append(training_set[sample_idx][properties.R])

        # the variances of all structures at once instead of one at a time
        sizes = [len(positions) for positions in structures]
        idx_m = torch.repeat_interleave(torch.arange(len(sizes)), torch.tensor(sizes))
        all_variances = batch_compute_principal_variances(torch.cat(structures), idx_m)

        grouped_variances: Dict[int, List[torch.Tensor]] = defaultdict(list)
        for n_atoms, variances in zip(sizes, all_variances):
            grouped_variances[n_atoms].append(variances)

        fit_iterator = tqdm(
            grouped_variances.items(),
            total=len(grouped_variances),
            desc="Fitting variance GMMs",
            disable=not self.verbose,
        )
        skipped: List[int] = []
        for n_atoms, variances in fit_iterator:
            # GaussianMixture needs at least two samples. Rare atom counts can
            # fall below that on small splits; skipping keeps the fit alive and
            # sample() reports the gap when one is actually requested.
            if len(variances) < 2:
                skipped.append(n_atoms)
                continue

            # Fit in log space so sampled variances are always positive. Planar
            # and linear molecules need the floor; their smallest variance is 0.
            variance_tensor = torch.stack(variances).clamp(min=self.variance_floor)
            log_variances = torch.log(variance_tensor).cpu().numpy()
            n_components = min(self.n_components, len(variances))
            model = GaussianMixture(n_components=n_components, covariance_type="full")
            model.fit(log_variances)
            self.models[n_atoms] = model

        if skipped:
            warnings.warn(
                f"No mixture fitted for atom counts {sorted(skipped)}: fewer than two training "
                f"structures each. Sampling a molecule of that size will raise.",
                stacklevel=2,
            )

    def sample(
        self,
        n_atoms: int,
        num_samples: int = 1,
        target_ratio: Optional[Sequence[float]] = None,
    ) -> torch.Tensor:
        """
        Sample principal-axis variances for molecules with `n_atoms` atoms.

        Args:
            n_atoms: Number of atoms; selects the per-size mixture.
            num_samples: Number of variance vectors to draw.
            target_ratio: Optional (xx, yy, zz) principal-axis variance ratio,
                normalized internally. When set, each sampled vector keeps its
                total variance (the molecule size the mixture predicts) but has
                that variance redistributed to match the requested shape.

        Returns:
            Tensor of shape (num_samples, 3), descending along the last axis.
        """
        if n_atoms not in self.models:
            raise ValueError(f"No shape model trained for n_atoms={n_atoms}. Available: {sorted(self.models)}")

        from sklearn.utils import check_random_state

        model = self.models[n_atoms]
        log_variances, _ = model.sample(num_samples)
        # GaussianMixture.sample returns the draws grouped by component; shuffle
        # them so that every row is an independent draw. One row needs no shuffle
        # and draws nothing from the RNG.
        log_variances = log_variances[check_random_state(model.random_state).permutation(num_samples)]
        variances = torch.exp(torch.tensor(log_variances, dtype=torch.float32))
        # The mixture cannot guarantee ordering; keep the descending convention.
        variances = variances.sort(dim=-1, descending=True).values
        if target_ratio is not None:
            variances = apply_target_ratio(variances, normalize_target_ratio(target_ratio))
        return variances

    def sample_from_batch(
        self,
        batch: Dict[str, torch.Tensor],
        target_ratio: Optional[Sequence[float]] = None,
    ) -> torch.Tensor:
        """
        Sample one variance vector per molecule in a batch.

        Args:
            batch: A schnetpack batch dict; molecule sizes are read from idx_m.
            target_ratio: Optional shape to prescribe for every molecule of the
                batch, see :meth:`sample`.

        Returns:
            Tensor of shape (n_molecules, 3), descending along the last axis,
            ordered like idx_m.unique(), on the same device and dtype as the
            batch positions.
        """
        positions = batch[properties.R]
        n_atoms = batch[properties.idx_m].unique(return_counts=True)[1].cpu()
        # one draw per molecule size rather than per molecule
        variances = torch.empty(len(n_atoms), 3)
        for size in n_atoms.unique().tolist():
            molecules = n_atoms == size
            variances[molecules] = self.sample(
                n_atoms=size,
                num_samples=int(molecules.sum()),
                target_ratio=target_ratio,
            )
        return variances.to(dtype=positions.dtype, device=positions.device)
