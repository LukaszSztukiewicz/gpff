"""Data-anchored one-step recovery augmentation, separate from self-conditioning."""
import torch
from schnetpack import properties as P
from gpff import keys
from gpff.analysis.absorbing_diagnostics import snapshot


class RecoveryAugmentation:
    def __init__(self, mode="rollout", probability=0.5, step_size=1.0,
                 extra_noise_std=0.0, seed=12345, weight_generator=None):
        if mode not in ("ordinary", "rollout", "isotropic"):
            raise ValueError("Unknown recovery mode")
        if not 0 <= probability <= 1 or not 0 <= step_size <= 1 or extra_noise_std < 0:
            raise ValueError("Invalid recovery parameters")
        self.mode, self.probability, self.step_size = mode, probability, step_size
        self.extra_noise_std, self.seed = extra_noise_std, seed
        self.weight_generator = weight_generator

    @torch.no_grad()
    def __call__(self, batch, teacher, epoch, batch_index):
        out = snapshot(batch)
        x = out[P.R]
        idx = out[P.idx_m]
        rng = torch.Generator(device=x.device).manual_seed(self.seed + 1000003 * epoch + batch_index)
        selected_m = torch.rand(len(out[P.n_atoms]), device=x.device, generator=rng) < self.probability
        selected = selected_m[idx]
        if self.mode == "ordinary":
            return out, {}
        teacher_input = snapshot(batch)
        delta = teacher(teacher_input)[keys.forces].detach() * (self.step_size / 2)
        # Remove translation without changing atom correspondence or matching.
        n_molecules = len(out[P.n_atoms])
        counts = out[P.n_atoms].to(x.dtype)
        means = x.new_zeros((n_molecules, 3)).index_add_(0, idx, delta) / counts[:, None]
        delta -= means[idx]
        if self.mode == "isotropic":
            noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=rng)
            noise -= (x.new_zeros((n_molecules,3)).index_add_(0,idx,noise) / counts[:,None])[idx]
            target_norm = x.new_zeros(n_molecules).index_add_(0,idx,delta.square().sum(-1)).sqrt()
            noise_norm = x.new_zeros(n_molecules).index_add_(0,idx,noise.square().sum(-1)).sqrt()
            noise *= (target_norm / noise_norm.clamp_min(1e-12))[idx,None]
            delta = noise
        if self.extra_noise_std:
            delta += self.extra_noise_std * torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=rng)
        if not bool(torch.isfinite(delta[selected]).all()):
            raise ValueError("Nonfinite teacher rollout; refusing corrupted labels")
        out[P.R] = x + delta * selected[:, None]
        # Keep the INITIAL corruption sigma, masking, clean labels and alignment.
        # This is augmentation, not a new diffusion bound or residual sigma.
        out[keys.forces] = 2 * (out[keys.x_0] - out[P.R])
        squared = (out[keys.x_0] - out[P.R]).square().sum(-1)
        energy = x.new_zeros(len(out[P.n_atoms]))
        energy.index_add_(0, idx, squared)
        out[keys.energy] = energy.reshape_as(out[keys.energy])
        if self.weight_generator is not None:
            sigma = out[keys.sigma_curr].reshape(-1,1)
            if len(sigma) != len(idx):
                sigma = sigma[idx]
            ew,fw = self.weight_generator(sigma=sigma,x_0=out[keys.x_0],x_t=out[P.R],
                energy=energy,forces=out[keys.forces])
            starts = torch.cumsum(out[P.n_atoms],0)-out[P.n_atoms]
            if ew.numel()==len(idx):
                ew = ew.reshape(-1)[starts]
            out[keys.energy_weight], out[keys.forces_weight] = ew.reshape_as(out[keys.energy]),fw
        displacement = torch.linalg.vector_norm(delta[selected], dim=-1)
        error = torch.linalg.vector_norm(out[P.R][selected] - out[keys.x_0][selected], dim=-1)
        selected_indices = torch.where(selected_m)[0]
        displacement_rms = (x.new_zeros(n_molecules).index_add_(0,idx,delta.square().sum(-1))/(3*counts)).sqrt()
        error_rms = (energy/(3*counts)).sqrt()
        sigmas=out[keys.sigma_curr].reshape(-1)
        if len(sigmas)==len(idx):
            sigmas=sigmas[torch.cumsum(out[P.n_atoms],0)-out[P.n_atoms]]
        values=torch.stack([selected_indices.to(x.dtype),sigmas[selected_indices],
            displacement_rms[selected_indices],error_rms[selected_indices]],dim=-1).cpu().tolist()
        self.last_records=[dict(zip(("molecule","initial_sigma","displacement_rms","rollout_error_rms"),row)) for row in values]
        metrics = {"fraction": selected_m.float().mean()}
        if displacement.numel():
            metrics.update(displacement_mean=displacement.mean(), displacement_p90=torch.quantile(displacement, 0.9),
                           error_mean=error.mean(), error_p90=torch.quantile(error, 0.9))
        return out, metrics
