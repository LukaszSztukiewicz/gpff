import torch
from scipy.optimize import linear_sum_assignment
import numpy as np
from typing import List, Dict, Tuple, Optional
import warnings
from ase.build import minimize_rotation_and_translation
from ase import Atoms
from schnetpack import properties
from gpff import keys


__all__ = [
    "compute_covariance_matrix",
    "align_to_principal_components",
    "batch_align_to_principal_components",
    "sample_covariance_noise",
    "apply_covariance",
    "batch_apply_covariances",
    "batch_compute_principal_variances",
    "batch_match_principal_variances",
    "align_rotation",
    "align_permutation",
    "align",
]


def compute_covariance_matrix(positions, numbers=None, ignore_hydrogens=False, epsilon=1e-6):
    """
    Get the covariance matrix of the positions.
    
    Args:
        positions: Tensor or array of shape (n_atoms, 3) containing atomic positions.
        numbers: Optional atomic numbers for filtering.
        ignore_hydrogens: If True, ignore hydrogen atoms when computing covariance.
        epsilon: Small value added to diagonal for numerical stability (default: 1e-6).
    
    Returns:
        Covariance matrix of shape (3, 3).
    """
    numpy_inputs = False
    if isinstance(positions, np.ndarray):
        positions = torch.from_numpy(positions)
        numpy_inputs = True

    positions = positions.clone()
    if ignore_hydrogens and numbers is not None:
        positions = positions[numbers != 1]

    # Center positions
    centroid = positions.mean(dim=0)
    positions = positions - centroid
    
    # Handle edge case: if only one position, return identity matrix scaled by epsilon
    n_positions = len(positions)
    if n_positions <= 1:
        covariance = torch.eye(3, device=positions.device, dtype=positions.dtype) * epsilon
    else:
        covariance = positions.T @ positions / (n_positions - 1) # -1 for Bessel's correction
        covariance += torch.eye(3, device=covariance.device, dtype=covariance.dtype) * epsilon
        
    if numpy_inputs:
        covariance = covariance.numpy()
    return covariance


def align_to_principal_components(positions, numbers=None, ignore_hydrogens=False):
    """
    Align the x, y, z coordinates of the positions to the principal components of the covariance matrix.
    Removes the centroid of the positions.
    """
    if ignore_hydrogens and numbers is None:
        raise ValueError("numbers must be provided if ignore_hydrogens is True")

    numpy_inputs = False
    if isinstance(positions, np.ndarray):
        positions = torch.from_numpy(positions)
        numpy_inputs = True

    # Center positions
    centered_positions = positions - positions.mean(dim=0, keepdim=True)

    cov_positions = centered_positions.clone()
    # get covariance matrix
    if ignore_hydrogens:
        cov_positions = cov_positions[numbers != 1]
    # Pass numbers=None since we've already filtered, and ignore_hydrogens=False since filtering is done
    covariance = compute_covariance_matrix(cov_positions, numbers=None, ignore_hydrogens=False)

    # PCA
    eigen_vals, eigen_vecs = torch.linalg.eigh(covariance)
    order = torch.argsort(eigen_vals, descending=True)
    eigen_vecs = eigen_vecs[:, order]
    # flip if necessary
    if torch.linalg.det(eigen_vecs) < 0:
        eigen_vecs[:, -1] *= -1  

    # Rotate centered positions
    aligned_centered = centered_positions @ eigen_vecs

    # convert back to numpy if necessary
    if numpy_inputs:
        aligned_centered = aligned_centered.numpy()
    return aligned_centered


def batch_align_to_principal_components(
    property_values: torch.Tensor,
    idx_m: torch.Tensor,
) -> torch.Tensor:
    """
    Align each molecule independently to its principal components while preserving
    the original molecular centroid.
    """
    aligned_property = property_values.clone()

    for mol_idx in idx_m.unique():
        mol_mask = idx_m == mol_idx
        mol_property = property_values[mol_mask]
        centroid = mol_property.mean(dim=0, keepdim=True)
        aligned_property[mol_mask] = align_to_principal_components(mol_property) + centroid

    return aligned_property
    

def sample_covariance_noise(positions, add_identity: bool = True):
    """
    Sample noise from a multivariate normal distribution with the same covariance matrix as the positions.
    NOTE: We add the identity matrix to the covariance matrix to ensure that the noise non-zero in all directions.
    """
    covariance = compute_covariance_matrix(positions)
    if add_identity:
        covariance += torch.eye(positions.shape[1], device=positions.device, dtype=positions.dtype)
    mean = torch.zeros(positions.shape[1], device=positions.device, dtype=positions.dtype)
    
    # Use torch multivariate normal sampling
    mvn = torch.distributions.MultivariateNormal(mean, covariance)
    noise = mvn.sample((positions.shape[0],))
    return noise


def apply_covariance(positions: torch.Tensor, target_covariance: torch.Tensor) -> torch.Tensor:
    """
    Transform positions to have the specified target covariance matrix.
    
    Args:
        positions: Tensor of shape (n_atoms, 3) containing atomic positions.
        target_covariance: Tensor of shape (3, 3) containing the target covariance matrix.
        
    Returns:
        Transformed positions with the target covariance.
    """
    numpy_inputs = False
    if isinstance(positions, np.ndarray):
        positions = torch.from_numpy(positions)
        numpy_inputs = True
    if isinstance(target_covariance, np.ndarray):
        target_covariance = torch.from_numpy(target_covariance)
    
    # Center positions before computing covariance and applying transformation
    centroid = positions.mean(dim=0, keepdim=True)
    centered_positions = positions - centroid
    
    # Compute current covariance of centered positions
    current_covariance = compute_covariance_matrix(centered_positions)
    
    # Compute Cholesky decompositions
    L_target = torch.linalg.cholesky(target_covariance)
    L_current = torch.linalg.cholesky(current_covariance)
    
    # Compute inverse of current Cholesky factor
    L_current_inv = torch.linalg.solve_triangular(
        L_current, 
        torch.eye(3, device=L_current.device, dtype=L_current.dtype), 
        upper=False
    )
    
    # Create transformation matrix
    transform = L_target @ L_current_inv
    
    # Apply transformation to centered positions, then restore centroid
    transformed_centered = centered_positions @ transform.T
    transformed_positions = transformed_centered + centroid
    
    if numpy_inputs:
        transformed_positions = transformed_positions.numpy()
    
    return transformed_positions


def batch_apply_covariances(
    positions: torch.Tensor,
    idx_m: torch.Tensor,
    covariances: torch.Tensor,
    centers: Optional[torch.Tensor] = None,
    molecule_ids: Optional[torch.Tensor] = None,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """
    Transform each molecule to match a target covariance and optionally a target center.

    Args:
        positions: Position tensor of shape (n_atoms, 3).
        idx_m: Molecule index tensor of shape (n_atoms,) mapping each atom to its molecule.
        covariances: Covariance tensor of shape (n_molecules, 3, 3), ordered like molecule_ids.
        centers: Optional centroid tensor of shape (n_molecules, 3), ordered like
            molecule_ids. When given, each molecule is shifted to its target center;
            otherwise the current centroid is preserved.
        molecule_ids: Molecule ids the covariance rows refer to. Defaults to
            idx_m.unique(); pass explicitly when positions are a subset of a batch
            (e.g. free atoms only) so the rows stay aligned with the full batch.
        epsilon: Small value added to the covariance diagonal for numerical stability.

    Returns:
        Transformed position tensor of shape (n_atoms, 3).
    """
    if molecule_ids is None:
        molecule_ids = idx_m.unique()

    transformed = positions.clone()
    for row, mol_idx in enumerate(molecule_ids):
        mol_mask = idx_m == mol_idx
        pos = positions[mol_mask]
        if pos.shape[0] <= 1:
            continue

        covariance = covariances[row] + torch.eye(3, device=positions.device, dtype=positions.dtype) * epsilon
        try:
            mol_positions = apply_covariance(pos, covariance)
        except RuntimeError as e:
            warnings.warn(
                f"Failed to apply covariance transformation for molecule {int(mol_idx)}: {e}. "
                f"Keeping original positions.",
                RuntimeWarning,
                stacklevel=2
            )
            continue
        if centers is not None:
            mol_positions = mol_positions - mol_positions.mean(dim=0) + centers[row]
        transformed[mol_mask] = mol_positions

    return transformed


def batch_compute_principal_variances(
    positions: torch.Tensor,
    idx_m: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the principal variances of each molecule.

    Args:
        positions: Position tensor of shape (n_atoms, 3).
        idx_m: Molecule index tensor of shape (n_atoms,) mapping each atom to its molecule.

    Returns:
        Variance tensor of shape (n_molecules, 3), descending along the last
        axis, ordered like idx_m.unique().
    """
    variances = []
    for mol_idx in idx_m.unique():
        pos = positions[idx_m == mol_idx]
        pos = pos - pos.mean(dim=0, keepdim=True)
        variances.append(torch.linalg.eigvalsh(pos.T @ pos / len(pos)).flip(0))
    return torch.stack(variances)


def _robust_symeig(cov: torch.Tensor, epsilon: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Eigendecomposition of a small symmetric matrix that tolerates the
    ill-conditioned / near-degenerate covariances (repeated principal variances)
    that make the accelerator's ``eigh`` fail to converge.

    Solves in float64 with a trace-scaled diagonal jitter for conditioning, and
    retries on CPU if the accelerator's solver still fails to converge. Returns
    ascending eigenvalues and eigenvectors in the input's dtype and device.
    """
    dim = cov.shape[-1]
    cov64 = cov.double()
    jitter = (epsilon * cov64.diagonal(dim1=-2, dim2=-1).abs().mean().clamp(min=epsilon))
    cov64 = cov64 + jitter * torch.eye(dim, dtype=cov64.dtype, device=cov64.device)
    try:
        lam, axes = torch.linalg.eigh(cov64)
    except torch._C._LinAlgError:
        try:
            lam, axes = torch.linalg.eigh(cov64.cpu())
            lam, axes = lam.to(cov.device), axes.to(cov.device)
        except torch._C._LinAlgError:
            # A tiny symmetric matrix that neither solver will decompose should
            # not abort a whole sampling run. Fall back to the diagonal with an
            # identity basis: exact when the axes are already aligned, and a
            # harmless approximation otherwise. Callers guard non-finite inputs
            # before reaching here, so this only covers pathological conditioning.
            lam = cov64.diagonal(dim1=-2, dim2=-1).clamp(min=epsilon)
            axes = torch.eye(dim, dtype=cov64.dtype, device=cov64.device)
    return lam.to(cov.dtype), axes.to(cov.dtype)


def batch_match_principal_variances(
    positions: torch.Tensor,
    target_variances: torch.Tensor,
    idx_m: torch.Tensor,
    preserve_spread: bool = False,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Rescale each molecule along its own principal axes so that its principal
    variances match the target. Orientation and centroid of each molecule are
    preserved.

    Args:
        positions: Position tensor of shape (n_atoms, 3) to reshape.
        target_variances: Variance tensor of shape (n_molecules, 3), descending
            along the last axis, ordered like idx_m.unique().
        idx_m: Molecule index tensor of shape (n_atoms,) mapping each atom to its molecule.
        preserve_spread: If True, only the ratios of the target variances are
            enforced and each molecule keeps its current total variance.
        epsilon: Small value to guard divisions for degenerate molecules.

    Returns:
        Reshaped position tensor of shape (n_atoms, 3).
    """
    reshaped = positions.clone()

    for mol_position, mol_idx in enumerate(idx_m.unique()):
        mol_mask = idx_m == mol_idx
        pos = positions[mol_mask]
        if pos.shape[0] <= 1:
            continue

        centroid = pos.mean(dim=0, keepdim=True)
        pos = pos - centroid

        cov = pos.T @ pos / len(pos)
        # A molecule whose denoising diverged carries non-finite positions, so
        # its covariance is non-finite and eigh cannot converge. Leave it
        # untouched — it stays non-finite and is dropped as invalid downstream —
        # rather than aborting the reshape (and the whole run) for the batch. The
        # strongly elongated targets (e.g. a [0.9, 0.05, 0.05] rod) are what
        # occasionally push a molecule there.
        if not torch.isfinite(cov).all():
            continue

        # eigh returns ascending eigenvalues; flip both to match the descending target.
        lam, axes = _robust_symeig(cov, epsilon)
        lam, axes = lam.flip(0), axes.flip(1)

        target = target_variances[mol_position]
        if preserve_spread:
            target = target / target.sum().clamp(min=epsilon) * lam.sum()

        # Per-axis factor turning each current variance into its target one;
        # sqrt because positions scale with the standard deviation.
        scale = (target / lam.clamp(min=epsilon)).sqrt()

        # Rotate into the principal frame, scale there, rotate back.
        reshaped[mol_mask] = ((pos @ axes) * scale) @ axes.T + centroid

    return reshaped


def align_rotation(positions: np.ndarray | torch.Tensor, target_positions: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """
    Wrapper for ase.build.minimize_rotation_and_translation.
    """
    # convert torch to numpy if necessary
    torch_inputs = False
    if isinstance(positions, torch.Tensor):
        dtype = positions.dtype
        device = positions.device
        positions = positions.numpy()
        target_positions = target_positions.numpy()
        torch_inputs = True

    # build the atoms objects
    numbers = np.ones(positions.shape[0])
    atoms = Atoms(positions=positions, numbers=numbers)
    target_atoms = Atoms(positions=target_positions, numbers=numbers)
    
    # align the positions
    minimize_rotation_and_translation(target_atoms, atoms)
    positions = atoms.positions

    if torch_inputs:
        positions = torch.from_numpy(positions)
        positions = positions.to(dtype=dtype, device=device)

    return positions


def align_permutation(
    positions: np.ndarray | torch.Tensor,
    target_positions: np.ndarray | torch.Tensor,
    numbers: np.ndarray | torch.Tensor,
    target_numbers: np.ndarray | torch.Tensor,
) -> Tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
    """
    Find the best permutation of positions to align with target_positions using the Hungarian algorithm.
    
    Args:
        positions: (N, 3) tensor of noisy coordinates to align.
        target_positions: (N, 3) tensor of reference coordinates.
        numbers: (N,) tensor of atomic numbers for positions.
        target_numbers: (N,) tensor of atomic numbers for target_positions.
    
    Returns:
        Tuple of (aligned_positions, aligned_numbers) where aligned_positions is the permuted
        positions that best match target_positions, and aligned_numbers are the corresponding
        permuted atomic numbers.
    """
    # Convert numpy inputs to torch
    numpy_inputs = False
    if isinstance(positions, np.ndarray):
        positions = torch.from_numpy(positions)
        target_positions = torch.from_numpy(target_positions)
        numbers = torch.from_numpy(numbers)
        target_numbers = torch.from_numpy(target_numbers)
        numpy_inputs = True

    # Clone to avoid modifying inputs
    positions = positions.clone()
    numbers = numbers.clone()
    target_positions = target_positions.clone()

    # Align atomic numbers first (permute positions to match target_numbers)
    positions, numbers = align_numbers(positions, numbers, target_numbers)

    # Center both sets of positions (remove global translation)
    centroid = positions.mean(dim=0, keepdim=True)
    positions_centered = positions - centroid
    target_positions_centered = target_positions - target_positions.mean(dim=0, keepdim=True)

    # Compute cost matrix: pairwise squared distances between all atoms
    # Shape: (N, N) where cost_matrix[i, j] = ||target[i] - position[j]||^2
    cost_matrix = (target_positions_centered[:, None, :] - positions_centered[None, :, :]).norm(dim=-1)**2

    # Disallow mismatched atom types by setting a huge cost
    mismatch = numbers.unsqueeze(1) != target_numbers.unsqueeze(0)
    cost_matrix[mismatch] = 1e12

    # Solve assignment problem using Hungarian algorithm
    cost_matrix_np = cost_matrix.detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_matrix_np)
    # col_ind[i] gives the index in positions that should map to target_positions[i]
    permutation = col_ind

    # Apply permutation and restore centroid
    aligned_positions = positions_centered[permutation] + centroid
    aligned_numbers = numbers[permutation]

    # Convert back to numpy if needed
    if numpy_inputs:
        aligned_positions = aligned_positions.numpy()
        aligned_numbers = aligned_numbers.numpy()

    return aligned_positions, aligned_numbers


def align(
    positions: np.ndarray | torch.Tensor,
    target_positions: np.ndarray | torch.Tensor,
    numbers: np.ndarray | torch.Tensor,
    target_numbers: np.ndarray | torch.Tensor,
) -> Tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
    """
    Align the positions and numbers to the target positions and numbers.
    """
    positions, numbers = align_permutation(positions, target_positions, numbers, target_numbers)
    positions = align_rotation(positions, target_positions)
    
    return positions, numbers


def align_numbers(
    positions: np.ndarray | torch.Tensor,
    numbers: np.ndarray | torch.Tensor,
    target_numbers: np.ndarray | torch.Tensor
) -> Tuple[np.ndarray | torch.Tensor]:
    """
    Finds the permutation array (indices) 'I' such that permuted[I[k]] == target[k].

    This function is designed to handle duplicate values correctly.

    Args:
        target: The desired final array (e.g., [1, 1, 6, 1, 8]).
        permuted: The array that needs to be reordered (e.g., [6, 1, 1, 8, 1]).

    Returns:
        A list of indices (the permutation array) that aligns 'permuted' to 'target'.
    """
    numbers_list = numbers.tolist()
    target_numbers_list = target_numbers.tolist()
    # 1. Build a map (dictionary) from the 'permuted' array.
    # The key is the element value, and the value is a list (or queue) of 
    # the original indices where that element appears in 'permuted'.
    # e.g., {6: [0], 1: [1, 2, 4], 8: [3]}
    index_map: Dict[int, List[int]] = {}
    for i, val in enumerate(numbers_list):
        if val not in index_map:
            index_map[val] = []
        index_map[val].append(i)
        
    # 2. Construct the permutation array by iterating through the 'target' array.
    permutation_indices: List[int] = []
    
    for val_t in target_numbers_list:
        # Check if the value exists in the map (it should, as they are permutations).
        if val_t not in index_map or not index_map[val_t]:
            # This case should ideally not happen if 'permuted' is truly a permutation of 'target'.
            raise ValueError(f"Value {val_t} in 'target' not found (or all instances used) in 'permuted'. Arrays are not permutations.")
        
        # Get the next available original index for this value.
        # We use pop(0) to simulate a FIFO queue, ensuring we use indices 
        # in the order they appeared in 'permuted'.
        original_index = index_map[val_t].pop(0)
        
        # Add the index to our result list.
        permutation_indices.append(original_index)
    
    return positions[permutation_indices], numbers[permutation_indices]
