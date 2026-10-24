import pytest
import torch

from gpff.utils.alignment import align_permutation, apply_covariance, batch_apply_covariances


def per_molecule(positions, idx_m, covariances, centers, molecule_ids, epsilon=1e-6):
    """The per-molecule loop batch_apply_covariances replaces."""
    out = positions.clone()
    eye = torch.eye(3, dtype=positions.dtype)
    for row, molecule_id in enumerate(molecule_ids):
        mask = idx_m == molecule_id
        if mask.sum() <= 1:
            continue
        try:
            moved = apply_covariance(positions[mask], covariances[row] + eye * epsilon)
        except RuntimeError:
            continue
        if centers is not None:
            moved = moved - moved.mean(dim=0) + centers[row]
        out[mask] = moved
    return out


def inputs():
    g = torch.Generator().manual_seed(0)
    sizes = torch.tensor([6, 1, 9, 4, 12])
    idx_m = torch.repeat_interleave(torch.arange(len(sizes)), sizes)
    positions = torch.randn(len(idx_m), 3, generator=g, dtype=torch.float64)
    a = torch.randn(len(sizes), 3, 3, generator=g, dtype=torch.float64)
    covariances = a @ a.mT + 0.1 * torch.eye(3, dtype=torch.float64)
    centers = torch.randn(len(sizes), 3, generator=g, dtype=torch.float64)
    # the atoms of a molecule need not be contiguous
    order = torch.randperm(len(idx_m), generator=g)
    return positions[order], idx_m[order], covariances, centers


@pytest.mark.parametrize("with_centers", [True, False])
def test_matches_the_per_molecule_transform(with_centers):
    positions, idx_m, covariances, centers = inputs()
    centers = centers if with_centers else None

    out = batch_apply_covariances(positions, idx_m, covariances, centers=centers)

    expected = per_molecule(positions, idx_m, covariances, centers, idx_m.unique())
    torch.testing.assert_close(out, expected)


def test_rows_follow_molecule_ids_for_a_subset_of_atoms():
    positions, idx_m, covariances, centers = inputs()
    # molecule 0 keeps no atom, so row k of covariances is no longer molecule k of the subset
    free = idx_m != 0

    out = batch_apply_covariances(
        positions[free], idx_m[free], covariances, centers=centers, molecule_ids=idx_m.unique()
    )

    expected = per_molecule(positions[free], idx_m[free], covariances, centers, idx_m.unique())
    torch.testing.assert_close(out, expected)


def test_keeps_the_molecules_it_cannot_transform():
    positions, idx_m, covariances, centers = inputs()
    covariances[3] = -torch.eye(3, dtype=covariances.dtype)

    with pytest.warns(RuntimeWarning, match=r"molecules \[3\]"):
        out = batch_apply_covariances(positions, idx_m, covariances, centers=centers)

    assert torch.equal(out[idx_m == 3], positions[idx_m == 3])
    torch.testing.assert_close(out, per_molecule(positions, idx_m, covariances, centers, idx_m.unique()))


@pytest.mark.parametrize("permuted_numbers", [False, True])
def test_align_permutation_recovers_a_shuffled_copy(permuted_numbers):
    g = torch.Generator().manual_seed(0)
    target = torch.randn(12, 3, generator=g, dtype=torch.float64) * 3
    numbers = torch.tensor([6, 1, 1, 8, 6, 1, 7, 1, 6, 1, 1, 8])
    order = torch.randperm(12, generator=g)
    positions = target[order] + 0.01 * torch.randn(12, 3, generator=g, dtype=torch.float64)
    # the training transform passes the target numbers for both
    given_numbers = numbers[order] if permuted_numbers else numbers

    aligned, aligned_numbers = align_permutation(positions, target, given_numbers, numbers)

    assert torch.equal(aligned_numbers, numbers)
    if permuted_numbers:
        torch.testing.assert_close(aligned, target, atol=0.05, rtol=0)
