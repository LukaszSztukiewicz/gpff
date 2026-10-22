"""
Optimizers for the noise-free jumps of direct denoising.

Below transport_sigma, direct denoising iterates the jump x <- x + r with
r = x_0(x) - x = F / 2, the pseudo-PES force over 2. The jump is steepest descent
with the step that lands at the bottom of the ideal well, so every optimizer here
starts with that step (Adam aside) and differs in how it reuses earlier jumps:
Anderson combines them by least squares, the others are the usual geometry
optimizers. They act per molecule; the sampler hands them only noise-free jumps
and marks the first one after transport or noise as ``fresh``, which restarts them.
"""

# python
import math
import torch
from abc import ABC, abstractmethod
from typing import Dict, Tuple


__all__ = [
    "JumpOptimizer",
    "Anderson",
    "Momentum",
    "Nesterov",
    "MDMin",
    "FIRE",
    "ConjugateGradient",
    "Adam",
    "LBFGS",
]


def molecule_sum(values: torch.Tensor, idx_m: torch.Tensor, n_molecules: int) -> torch.Tensor:
    return torch.zeros(
        (n_molecules, *values.shape[1:]), device=values.device, dtype=values.dtype
    ).index_add_(0, idx_m, values)


def _dot(a: torch.Tensor, b: torch.Tensor, idx_m: torch.Tensor, n_molecules: int) -> torch.Tensor:
    """Per-molecule inner product of two (n_atoms, 3) tensors, per atom as (n_atoms, 1)."""
    return molecule_sum((a * b).sum(-1), idx_m, n_molecules)[idx_m].unsqueeze(-1)


def _cap(step: torch.Tensor, residual: torch.Tensor, max_step: float, idx_m, n_molecules) -> torch.Tensor:
    """Scale step per molecule to at most max_step times the plain jump."""
    ratio = (_dot(step, step, idx_m, n_molecules) / _dot(residual, residual, idx_m, n_molecules).clamp_min(
        torch.finfo(step.dtype).tiny
    )).sqrt()
    return step * (max_step / ratio.clamp_min(torch.finfo(step.dtype).tiny)).clamp(max=1)


class JumpOptimizer(ABC):
    """
    Base class. ``init_state`` gives the per-atom state tensors (leading dimension
    n_atoms), ``step`` the next positions from the current ones (the model input)
    and their residual r = x_0 - x.
    """

    @abstractmethod
    def init_state(self, positions: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def step(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        state: Dict[str, torch.Tensor],
        fresh: torch.Tensor,
        idx_m: torch.Tensor,
        n_molecules: int,
        last: bool,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            fresh: (n_atoms, 1) True where the history is invalid; restart there.
            last: this is the sampler's last step.
        """
        raise NotImplementedError


class Anderson(JumpOptimizer):
    """
    Anderson acceleration (DIIS): x_0 - sum_j gamma_j (x_0 - x_0_j), with gamma
    minimizing |r - sum_j gamma_j (r - r_j)| over the last ``depth`` jumps. Restarts
    when the residual grows; moves at most ``max_step`` times the plain jump beyond x_0.
    """

    def __init__(self, depth: int = 2, max_step: float = 10.0):
        if depth < 1:
            raise ValueError(f"depth must be positive, got {depth}")
        self.depth = depth
        self.max_step = max_step

    def init_state(self, positions):
        n_atoms = positions.shape[0]
        return {
            "x0": positions.new_zeros(n_atoms, self.depth, 3),
            "residual": positions.new_zeros(n_atoms, self.depth, 3),
            "length": torch.zeros(n_atoms, 1, dtype=torch.long, device=positions.device),
        }

    def step(self, x, residual, state, fresh, idx_m, n_molecules, last):
        x_0 = x + residual
        x0_history, residual_history = state["x0"], state["residual"]
        length = torch.where(fresh, 0, state["length"])
        grew = molecule_sum(residual.square().sum(-1), idx_m, n_molecules) > molecule_sum(
            residual_history[:, 0].square().sum(-1), idx_m, n_molecules
        )
        length = torch.where(grew[idx_m].unsqueeze(-1), 0, length)

        valid = (torch.arange(self.depth, device=length.device) < length).unsqueeze(-1)
        d_residual = (residual.unsqueeze(1) - residual_history) * valid
        d_x0 = (x_0.unsqueeze(1) - x0_history) * valid
        gram = molecule_sum(torch.einsum("aic,ajc->aij", d_residual, d_residual), idx_m, n_molecules).double()
        rhs = molecule_sum(torch.einsum("aic,ac->ai", d_residual, residual), idx_m, n_molecules).double()
        # empty entries get gamma = 0, a small relative ridge keeps the rest well posed
        diagonal = gram.diagonal(dim1=1, dim2=2)
        ridge = 1e-6 * diagonal.amax(-1, keepdim=True) + (diagonal == 0).double()
        gamma = torch.linalg.solve(gram + torch.diag_embed(ridge), rhs).to(x_0.dtype)
        correction = -torch.einsum("ai,aic->ac", gamma[idx_m], d_x0)

        # at most max_step times the plain jump
        jump_norm = molecule_sum(residual.square().sum(-1), idx_m, n_molecules).sqrt()
        correction_norm = molecule_sum(correction.square().sum(-1), idx_m, n_molecules).sqrt()
        scale = (self.max_step * jump_norm / correction_norm.clamp_min(torch.finfo(x_0.dtype).tiny)).clamp(max=1)
        accelerated = x_0 + scale[idx_m].unsqueeze(-1) * correction

        x_next = torch.where(length > 0, accelerated, x_0)
        return x_next, {
            "x0": torch.cat([x_0.unsqueeze(1), x0_history[:, :-1]], dim=1),
            "residual": torch.cat([residual.unsqueeze(1), residual_history[:, :-1]], dim=1),
            "length": (length + 1).clamp(max=self.depth),
        }


class Momentum(JumpOptimizer):
    """Heavy ball: v <- momentum v + r, x <- x + v."""

    def __init__(self, momentum: float = 0.5):
        self.momentum = momentum

    def init_state(self, positions):
        return {"v": torch.zeros_like(positions)}

    def step(self, x, residual, state, fresh, idx_m, n_molecules, last):
        v = self.momentum * torch.where(fresh, 0.0, state["v"]) + residual
        return x + v, {"v": v}


class Nesterov(JumpOptimizer):
    """
    Nesterov: the model sees the look-ahead y = x_k + momentum (x_k - x_k-1), and
    x_k+1 = y + r(y). The last step returns x_k+1 instead of the next look-ahead.
    """

    def __init__(self, momentum: float = 0.5):
        self.momentum = momentum

    def init_state(self, positions):
        return {"x": torch.zeros_like(positions)}

    def step(self, x, residual, state, fresh, idx_m, n_molecules, last):
        x_new = x + residual
        x_old = torch.where(fresh, x, state["x"])
        look_ahead = x_new + self.momentum * (x_new - x_old)
        return (x_new if last else look_ahead), {"x": x_new}


class MDMin(JumpOptimizer):
    """
    ASE's MDMin: velocity Verlet that keeps only the velocity along the force and
    none against it. dt = sqrt(2) makes the first step, dt^2 / 2 r, the plain jump.
    """

    def __init__(self, dt: float = math.sqrt(2)):
        self.dt = dt

    def init_state(self, positions):
        return {"v": torch.zeros_like(positions)}

    def step(self, x, residual, state, fresh, idx_m, n_molecules, last):
        v = state["v"] + 0.5 * self.dt * residual
        vf = _dot(v, residual, idx_m, n_molecules)
        ff = _dot(residual, residual, idx_m, n_molecules).clamp_min(torch.finfo(x.dtype).tiny)
        v = torch.where(fresh | (vf < 0), 0.0, residual * vf / ff)
        v = v + 0.5 * self.dt * residual
        return x + self.dt * v, {"v": v}


class FIRE(JumpOptimizer):
    """
    FIRE (Bitzek et al. 2006) as in ASE. dt = 1 makes the first step the plain jump,
    and dt_max = 1 keeps it the largest: a longer step overshoots the stiff modes,
    which the plain jump already puts at the bottom.
    """

    def __init__(
        self,
        dt: float = 1.0,
        dt_max: float = 1.0,
        n_min: int = 5,
        f_inc: float = 1.1,
        f_dec: float = 0.5,
        alpha_start: float = 0.1,
        f_alpha: float = 0.99,
    ):
        self.dt = dt
        self.dt_max = dt_max
        self.n_min = n_min
        self.f_inc = f_inc
        self.f_dec = f_dec
        self.alpha_start = alpha_start
        self.f_alpha = f_alpha

    def init_state(self, positions):
        ones = positions.new_ones(positions.shape[0], 1)
        return {"v": torch.zeros_like(positions), "dt": self.dt * ones, "alpha": self.alpha_start * ones,
                "n_downhill": 0 * ones}

    def step(self, x, residual, state, fresh, idx_m, n_molecules, last):
        v, dt, alpha, n_downhill = state["v"], state["dt"], state["alpha"], state["n_downhill"]
        vf = _dot(v, residual, idx_m, n_molecules)
        downhill = ~fresh & (vf > 0)
        uphill = ~fresh & ~downhill
        ramp = downhill & (n_downhill > self.n_min)

        tiny = torch.finfo(x.dtype).tiny
        v_norm = _dot(v, v, idx_m, n_molecules).sqrt()
        f_norm = _dot(residual, residual, idx_m, n_molecules).sqrt().clamp_min(tiny)
        mixed = (1 - alpha) * v + alpha * residual / f_norm * v_norm
        v = torch.where(downhill, mixed, 0.0)
        dt = torch.where(fresh, self.dt, torch.where(ramp, (dt * self.f_inc).clamp(max=self.dt_max),
                                                     torch.where(uphill, dt * self.f_dec, dt)))
        alpha = torch.where(fresh | uphill, self.alpha_start, torch.where(ramp, alpha * self.f_alpha, alpha))
        n_downhill = torch.where(downhill, n_downhill + 1, 0.0)

        v = v + dt * residual
        return x + dt * v, {"v": v, "dt": dt, "alpha": alpha, "n_downhill": n_downhill}


class ConjugateGradient(JumpOptimizer):
    """
    Linear conjugate gradient on the local linearization, one model call per step.
    The model sees the trial point x + d; the residual change r(x) - r(x + d) = A d
    gives the exact step along d (a secant line search) and the next conjugate
    direction. The residual at the new iterate is extrapolated, r - alpha A d, so it
    restarts from the trial point when the curvature d.Ad is not positive. The step
    along d is capped at max_step; the last step jumps from the iterate.

    (Polak-Ribiere CG with the unit step instead of a line search restarts at every
    step here: the residuals of successive jumps are not orthogonal.)
    """

    def __init__(self, max_step: float = 10.0):
        self.max_step = max_step

    def init_state(self, positions):
        return {"x": torch.zeros_like(positions), "residual": torch.zeros_like(positions),
                "direction": torch.zeros_like(positions)}

    def step(self, x, residual, state, fresh, idx_m, n_molecules, last):
        dot = lambda a, b: _dot(a, b, idx_m, n_molecules)
        tiny = torch.finfo(x.dtype).tiny
        x_iterate, r_iterate, direction = state["x"], state["residual"], state["direction"]

        a_direction = r_iterate - residual
        curvature = dot(direction, a_direction)
        rr = dot(r_iterate, r_iterate)
        line_search = ~fresh & (curvature > 0)
        alpha = (rr / curvature.clamp_min(tiny)).clamp(max=self.max_step)
        x_new = x_iterate + alpha * direction
        r_new = r_iterate - alpha * a_direction
        d_new = r_new + dot(r_new, r_new) / rr.clamp_min(tiny) * direction

        # restart from the trial point, the one the model saw
        x_new = torch.where(line_search, x_new, x)
        r_new = torch.where(line_search, r_new, residual)
        d_new = torch.where(line_search, d_new, residual)
        x_next = x_new + r_new if last else x_new + d_new
        return x_next, {"x": x_new, "residual": r_new, "direction": d_new}


class Adam(JumpOptimizer):
    """
    Adam along the force (gradient -r). Per coordinate, so not rotation-equivariant,
    and it moves by about lr per coordinate however small the force is.
    """

    def __init__(self, lr: float = 0.05, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def init_state(self, positions):
        return {"m": torch.zeros_like(positions), "v": torch.zeros_like(positions),
                "t": positions.new_zeros(positions.shape[0], 1)}

    def step(self, x, residual, state, fresh, idx_m, n_molecules, last):
        t = torch.where(fresh, 0.0, state["t"]) + 1
        m = self.beta1 * torch.where(fresh, 0.0, state["m"]) + (1 - self.beta1) * residual
        v = self.beta2 * torch.where(fresh, 0.0, state["v"]) + (1 - self.beta2) * residual.square()
        m_hat = m / (1 - self.beta1**t)
        v_hat = v / (1 - self.beta2**t)
        return x + self.lr * m_hat / (v_hat.sqrt() + self.eps), {"m": m, "v": v, "t": t}


class LBFGS(JumpOptimizer):
    """
    L-BFGS with the unit step instead of a line search and H0 = (s.y / y.y) I from
    the newest pair (I without one, which makes the first step the plain jump).
    Pairs without positive curvature are skipped, a direction that is not downhill
    falls back to the plain jump, and the step is capped at max_step times it.
    """

    def __init__(self, memory: int = 5, max_step: float = 10.0):
        if memory < 1:
            raise ValueError(f"memory must be positive, got {memory}")
        self.memory = memory
        self.max_step = max_step

    def init_state(self, positions):
        n_atoms = positions.shape[0]
        return {
            "s": positions.new_zeros(n_atoms, self.memory, 3),
            "y": positions.new_zeros(n_atoms, self.memory, 3),
            "length": torch.zeros(n_atoms, 1, dtype=torch.long, device=positions.device),
            "x": torch.zeros_like(positions),
            "residual": torch.zeros_like(positions),
        }

    def step(self, x, residual, state, fresh, idx_m, n_molecules, last):
        dot = lambda a, b: _dot(a, b, idx_m, n_molecules)
        tiny = torch.finfo(x.dtype).tiny
        length = torch.where(fresh, 0, state["length"])

        # the pair from the last step; the gradient is -r
        s_new, y_new = x - state["x"], state["residual"] - residual
        add = ~fresh & (dot(s_new, y_new) > 0)
        s = torch.where(add.unsqueeze(1), torch.cat([s_new.unsqueeze(1), state["s"][:, :-1]], dim=1), state["s"])
        y = torch.where(add.unsqueeze(1), torch.cat([y_new.unsqueeze(1), state["y"][:, :-1]], dim=1), state["y"])
        length = torch.where(add, (length + 1).clamp(max=self.memory), length)

        # two-loop recursion on the force, so the result is the step -H g
        q, alphas, rhos = residual, [], []
        for j in range(self.memory):
            valid = j < length
            rho = torch.where(valid, 1 / dot(y[:, j], s[:, j]).clamp_min(tiny), 0.0)
            alpha = rho * dot(s[:, j], q)
            q = q - alpha * y[:, j]
            alphas.append(alpha)
            rhos.append(rho)
        gamma = torch.where(length > 0, dot(s[:, 0], y[:, 0]) / dot(y[:, 0], y[:, 0]).clamp_min(tiny), 1.0)
        z = gamma * q
        for j in reversed(range(self.memory)):
            z = z + s[:, j] * (alphas[j] - rhos[j] * dot(y[:, j], z))

        z = torch.where(dot(z, residual) > 0, z, residual)
        z = _cap(z, residual, self.max_step, idx_m, n_molecules)
        return x + z, {"s": s, "y": y, "length": length, "x": x, "residual": residual}
