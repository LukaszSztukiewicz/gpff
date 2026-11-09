from typing import Final


t: Final = "t"
noise: Final = "eps"
eps_pred: Final = "eps_pred"
energy: Final = "energy"
forces: Final = "forces"
energy_weight: Final = "energy_weight"
forces_weight: Final = "forces_weight"
sigma_curr: Final = "sigma_curr"
sigma_predicted: Final = "sigma_predicted"
sigma_next: Final = "sigma_next"
sigma_last: Final = "sigma_last"
step_size: Final = "step_size"
score: Final = "score"
converged: Final = "converged"
x_0: Final = "x_0"
x_prior: Final = "x_prior"
variance: Final = "variance"
scaffold_mask: Final = "scaffold_mask"
target_variances: Final = "target_variances"
free_covariance: Final = "free_covariance"
free_center: Final = "free_center"
nfe: Final = "nfe"
# models that generate the atom types (gpff.types)
h: Final = "h"  # (n_atoms, vocabulary size) the type vectors the model reads
type_logits: Final = "type_logits"  # (n_atoms, n_classes) predicted clean classes
type_target: Final = "type_target"  # (n_atoms,) clean classes
type_weight: Final = "type_weight"  # (n_atoms, 1) weights of the type cross-entropy
