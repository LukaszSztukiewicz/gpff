"""AdamW with a controlled update size for the weight matrices of equivariant blocks.

An equivariant layer mixes channels with one matrix that is shared over the 2l + 1
components of a feature (in PaiNN: the 3 components of a vector). Adam rescales every
entry on its own, so one learning rate still gives these matrices very different step
sizes. ``BlockAdamW`` rescales the update of each such matrix ("block") so that the
learning rate is its size, or replaces it by Muon's, NorMuon's or SOAP's update.

Manolache & Niepert, "Equivariance Breaks the Learning Rate" (NeurIPS 2026 workshop).
Vyas et al., "SOAP: Improving and Stabilizing Shampoo using Adam" (ICLR 2025).
Li et al., "NorMuon: Making Muon more efficient and scalable" (2025).
Amsel et al., "The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm" (ICLR 2026).
"""
import math
from functools import partial
from typing import Iterable, List, Optional

import torch
from schnetpack.nn import GatedEquivariantBlock
from schnetpack.representation.painn import PaiNNInteraction, PaiNNMixing
from torch import Tensor, nn
from torch.optim import Optimizer

from gpff.nn.normalized import NormalizedDense

__all__ = ["BlockAdamW", "block_param_groups", "refuse_weight_norm"]

# the equivariant blocks, with the name of their layer acting on vector features (None: no such layer)
_BLOCKS = {PaiNNInteraction: None, PaiNNMixing: "mu_channel_mix", GatedEquivariantBlock: "mix_vectors"}
# the fused layers of the blocks, by name in the block: one weight whose rows stack the outputs of this many layers
# of the same size, which the forward pass splits into the roles of the layer
_ROLES = {"interatomic_context_net.1": 3, "intraatomic_context_net.1": 3, "mu_channel_mix": 2, "mix_vectors": 2}
# the RMS of an AdamW update entry, which Muon's update is scaled to
_ADAMW_RMS = 0.2

# The functions on matrices take one matrix or a batch of them (the last two dimensions are a matrix).


def rms_norm(update: Tensor) -> Tensor:
    return update.pow(2).mean(dim=(-2, -1)).sqrt()


def spectral_norm(update: Tensor) -> Tensor:
    """Schatten-16 norm: an upper bound of the largest singular value, at most rank**(1/16) times it
    (exact for rank one). Two squarings of the Gram matrix replace an SVD, which is too slow for wide matrices."""
    frobenius = torch.linalg.matrix_norm(update)
    # singular values of at most one cannot overflow in the 16th power
    unit = update / frobenius.clamp_min(torch.finfo(update.dtype).tiny)[..., None, None]
    gram = unit.mT @ unit if unit.shape[-2] >= unit.shape[-1] else unit @ unit.mT
    gram = gram @ gram
    gram = gram @ gram
    return frobenius * torch.linalg.matrix_norm(gram) ** (1 / 8)


# Muon's quintic (Jordan et al. 2024): the same polynomial a x + b x^3 + c x^5 at every iteration
_NEWTON_SCHULZ = (3.4445, -4.7750, 2.0315)
# Polar Express (Amsel et al., ICLR 2026, Implementation 1): the minimax-optimal quintic of each iteration for
# singular values in [1e-3, 1]; iterations past the list repeat its last polynomial, Newton-Schulz's
# (15 x - 10 x^3 + 3 x^5) / 8. Every polynomial but the last takes x / 1.01, their safety factor for rounding.
_POLAR_EXPRESS = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),
]
_POLAR_EXPRESS = [(a / 1.01, b / 1.01**3, c / 1.01**5) for a, b, c in _POLAR_EXPRESS[:-1]] + _POLAR_EXPRESS[-1:]


def _odd_quintics(matrix: Tensor, coefficients, margin: float = 1.0) -> Tensor:
    """The quintics ``coefficients`` applied in turn to the singular values of ``matrix``, divided first by its
    Frobenius norm times ``margin``, so that they are at most one."""
    single = matrix.ndim == 2
    x = matrix[None] if single else matrix
    x = x / (torch.linalg.matrix_norm(x, keepdim=True) * margin).clamp_min(1e-7)
    tall = x.shape[-2] > x.shape[-1]
    if tall:
        x = x.mT
    for a, b, c in coefficients:
        gram = x @ x.mT
        # a * x + (b * gram + c * gram @ gram) @ x, as two fused matrix products
        x = torch.baddbmm(x, torch.baddbmm(gram, gram, gram, beta=b, alpha=c), x, beta=a)
    x = x.mT if tall else x
    return x[0] if single else x


def newton_schulz(matrix: Tensor, steps: int = 5) -> Tensor:
    """Muon's orthogonalization: the matrix with the singular vectors of ``matrix`` and singular values near one."""
    return _odd_quintics(matrix, [_NEWTON_SCHULZ] * steps)


def polar_express(matrix: Tensor, steps: int = 5) -> Tensor:
    """Polar Express's orthogonalization: as ``newton_schulz``, with the singular values closer to one."""
    coefficients = _POLAR_EXPRESS[:steps] + _POLAR_EXPRESS[-1:] * max(steps - len(_POLAR_EXPRESS), 0)
    return _odd_quintics(matrix, coefficients, margin=1.01)


_NORMS = {"rms": rms_norm, "spectral": spectral_norm}
_ORTHOGONALIZATIONS = {"newton_schulz": newton_schulz, "polar_express": polar_express}


def normalized(update: Tensor, norm: str) -> Tensor:
    """``update`` divided by its RMS or spectral norm (zero stays zero)."""
    return update / _NORMS[norm](update).clamp_min(torch.finfo(update.dtype).tiny)[..., None, None]


def orthogonalized(matrix: Tensor, method: str = "newton_schulz") -> Tensor:
    """Muon's update of a momentum matrix: orthogonalized by ``method``, scaled to the RMS of an AdamW update."""
    return _ORTHOGONALIZATIONS[method](matrix) * (_ADAMW_RMS * math.sqrt(max(matrix.shape[-2:])))


def row_normalized(matrix: Tensor, second_moment: Tensor, beta: float, method: str = "newton_schulz") -> Tensor:
    """NorMuon's update of a momentum matrix: orthogonalized by ``method``, every row divided by the root of its
    running mean square (``second_moment``, one per row, updated in place with decay ``beta``), then scaled back to
    the norm it had before, so that its size is Muon's (as in the authors' code)."""
    update = _ORTHOGONALIZATIONS[method](matrix)
    second_moment.lerp_(update.pow(2).mean(dim=-1, keepdim=True), 1 - beta)
    rows = update / (second_moment.sqrt() + 1e-10)
    norm = partial(torch.linalg.matrix_norm, keepdim=True)
    rows = rows * (norm(update) / (norm(rows) + 1e-10))
    return rows * (_ADAMW_RMS * math.sqrt(max(matrix.shape[-2:])))


def roles_of(matrix: Tensor, roles: int) -> Tensor:
    """The ``roles`` equal slices of the rows of ``matrix``, as a batch of matrices."""
    return matrix.reshape(roles, matrix.shape[0] // roles, matrix.shape[1])


def _groups(keys) -> List[List[int]]:
    """The indices of ``keys`` grouped by equal key, in order of first appearance."""
    groups = {}
    for i, key in enumerate(keys):
        groups.setdefault(key, []).append(i)
    return list(groups.values())


def _batched(function, matrices: List[Tensor], roles: int, states: Optional[List[Tensor]] = None) -> List[Tensor]:
    """``function`` of every role of every matrix on its own, as one call per shape: the results, with the shapes
    of ``matrices``. With ``states`` (one tensor per matrix with a row per row of it), ``function`` also takes
    their batch and updates it in place, and they take its values."""
    results = [None] * len(matrices)
    for indices in _groups(m.shape for m in matrices):
        inputs = [torch.cat([roles_of(matrices[i], roles) for i in indices])]
        if states is not None:
            inputs.append(torch.cat([roles_of(states[i], roles) for i in indices]))
        batch = function(*inputs)
        if states is not None:
            parts = [s.reshape(states[i].shape) for i, s in zip(indices, inputs[1].split(roles))]
            torch._foreach_copy_([states[i] for i in indices], parts)
        for i, result in zip(indices, batch.split(roles)):
            results[i] = result.reshape(matrices[i].shape)
    return results


def eigenbasis(gram: Tensor) -> Tensor:
    """The eigenvectors of a symmetric matrix as columns, by decreasing eigenvalue."""
    return torch.linalg.eigh(gram).eigenvectors.flip(-1)


def refreshed_basis(gram: Tensor, basis: Tensor, second_moment: Tensor, dim: int):
    """One step of power iteration from ``basis`` towards the eigenbasis of ``gram``: what SOAP does between two
    eigendecompositions, which is cheaper. The columns of ``basis`` are first sorted by their eigenvalue
    estimate, and ``second_moment`` (whose matrix axis ``dim``, 0 or 1, runs along them) with them, so that each of
    its entries stays with its direction when the eigenvalues change their order. Returns the new basis and the
    sorted second moment."""
    estimate = (basis * (gram @ basis)).sum(dim=-2)
    order = estimate.argsort(dim=-1, descending=True)
    basis = basis.gather(-1, order[..., None, :].expand_as(basis))
    index = order[..., :, None] if dim == 0 else order[..., None, :]
    return torch.linalg.qr(gram @ basis).Q, second_moment.gather(dim - 2, index.expand_as(second_moment))


def _rotate(x: Tensor, left: Optional[Tensor], right: Optional[Tensor]) -> Tensor:
    """``left.T @ x @ right``, into the eigenbases of SOAP; a side without a basis is not rotated."""
    if left is not None:
        x = left.mT @ x
    return x if right is None else x @ right


def _rotate_back(x: Tensor, left: Optional[Tensor], right: Optional[Tensor]) -> Tensor:
    """``left @ x @ right.T``, out of the eigenbases of SOAP."""
    if left is not None:
        x = left @ x
    return x if right is None else x @ right.mT


# the Gram matrices of SOAP's sides, of a batch of gradients
_SIDES = {"left": lambda grad: grad @ grad.mT, "right": lambda grad: grad.mT @ grad}


class BlockAdamW(Optimizer):
    """AdamW; the matrices of parameter groups with ``block=True`` (see ``block_param_groups``) take
    ``block_update`` steps instead:

    * ``"adam"``: AdamW's.
    * ``"rms"``, ``"spectral"``: AdamW's update divided by its RMS or spectral norm, so that the step has
      exactly this norm times ``lr`` and the direction of AdamW's. Single learning rate for all blocks;
      zero updates stay zero.
    * ``"muon"``: Muon's update, Nesterov ``momentum`` and Newton-Schulz orthogonalization (no Adam moments),
      scaled to the RMS of an AdamW update, so ``lr`` is comparable to AdamW's.
      ``orthogonalization`` is ``"newton_schulz"`` (Muon's, the default) or ``"polar_express"`` (Amsel et al.:
      a polynomial per iteration, singular values closer to one for the same 5 iterations).
    * ``"normuon"``: NorMuon (Li et al. 2025), Muon's update with every row (output neuron) divided by the root of
      its running mean square (decay ``row_beta``), then scaled back to Muon's size: no neuron dominates the step.
    * ``"soap"``: Adam in the eigenbasis of Shampoo's preconditioner, as in the reference implementation of
      SOAP. Every matrix keeps the averages (``shampoo_beta``, default ``betas[1]``) of ``G G^T`` and ``G^T G``
      over its gradients ``G``, and Adam's moments are those of ``G`` rotated into their eigenvectors, which are
      refreshed every ``precondition_frequency`` steps by one power iteration. The rotation keeps the size of the
      update, so ``lr`` is AdamW's. The first gradient only starts the preconditioner: no step, no decay.
      ``one_sided=True`` rotates only the smaller side of every matrix (the right one if they are equal), as in
      the one-sided variant of the SOAP paper; the other side keeps the coordinates of the weight. The refresh
      costs cubic time in the rotated sides.

    ``weight_decay`` is decoupled, as in AdamW: a step shrinks the weights by ``lr * weight_decay`` before the
    update, whichever update it is. It is off by default; the groups of ``block_param_groups`` decay the block
    matrices only.

    A block group may set ``roles`` (default 1) for fused layers: each of its weights stacks that many equal
    matrices along its rows, one per role of the layer's output. Every block update then takes the step of each
    of them on its own, as if the roles were separate layers: the size of a role's step does not depend on the
    gradients of the other roles, and SOAP preconditions each role with its own bases. ``"adam"`` is entrywise,
    so roles do not change it.

    A network has a few dozen small block matrices, so a step is bound by the launch of GPU kernels, not by
    their arithmetic. The step therefore works on lists of tensors (``torch._foreach_*``) and runs the matrix
    operations of all matrices (and roles) of one shape as one batched call. The parameters of a block group have
    to share their dtype and device.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        block_update: str = "rms",
        momentum: float = 0.95,
        precondition_frequency: int = 10,
        shampoo_beta: Optional[float] = None,
        one_sided: bool = False,
        orthogonalization: str = "newton_schulz",
        row_beta: float = 0.95,
    ):
        if block_update not in ("adam", "muon", "normuon", "soap", *_NORMS):
            raise ValueError(f"unknown block_update {block_update!r}")
        if orthogonalization not in _ORTHOGONALIZATIONS:
            raise ValueError(f"unknown orthogonalization {orthogonalization!r}")
        if not 0 <= row_beta < 1:
            raise ValueError("row_beta has to be in [0, 1)")
        if precondition_frequency < 1:
            raise ValueError("precondition_frequency has to be at least 1")
        if shampoo_beta is not None and not 0 <= shampoo_beta < 1:
            raise ValueError("shampoo_beta has to be in [0, 1)")
        defaults = dict(
            lr=lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay,
            block=False, block_update=block_update, momentum=momentum, roles=1,
            precondition_frequency=precondition_frequency, shampoo_beta=shampoo_beta, one_sided=one_sided,
            orthogonalization=orthogonalization, row_beta=row_beta,
        )
        super().__init__(params, defaults)

    def add_param_group(self, param_group):
        super().add_param_group(param_group)
        group = self.param_groups[-1]
        if group["block"] and any(p.ndim != 2 for p in group["params"]):
            raise ValueError("the parameters of a block group have to be weight matrices")
        if group["roles"] < 1 or (group["block"] and any(p.shape[0] % group["roles"] for p in group["params"])):
            raise ValueError("the rows of a block matrix have to split into the roles of its group")
        if group["block"] and len({(p.dtype, p.device) for p in group["params"]}) > 1:
            raise ValueError("the parameters of a block group have to share their dtype and device")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            mode = group["block_update"] if group["block"] else "adam"
            if mode in ("muon", "normuon"):
                updates = self._muon_updates(params, group, rows=mode == "normuon")
            elif mode == "soap":
                updates = self._soap_updates(params, group)
            else:
                updates = self._adam_updates(params, group)
                if mode in _NORMS:
                    updates = _batched(lambda batch: normalized(batch, mode), updates, group["roles"])
            # SOAP's first gradient of a matrix only starts its preconditioner: no step, no decay
            stepped = [i for i, update in enumerate(updates) if update is not None]
            if not stepped:
                continue
            params, updates = [params[i] for i in stepped], [updates[i] for i in stepped]
            if group["weight_decay"]:
                torch._foreach_mul_(params, 1 - group["lr"] * group["weight_decay"])
            torch._foreach_add_(params, updates, alpha=-group["lr"])
        return loss

    def _adam_updates(self, params: List[Tensor], group: dict) -> List[Tensor]:
        states = [self.state[p] for p in params]
        for p, state in zip(params, states):
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
        if not params:
            return []
        beta1, beta2 = group["betas"]
        grads = [p.grad for p in params]
        exp_avgs, exp_avg_sqs = [s["exp_avg"] for s in states], [s["exp_avg_sq"] for s in states]
        torch._foreach_lerp_(exp_avgs, grads, 1 - beta1)
        torch._foreach_mul_(exp_avg_sqs, beta2)
        torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1 - beta2)
        # the bias corrections of every parameter's own step (a parameter without a gradient falls behind)
        denominators = torch._foreach_div(exp_avg_sqs, [1 - beta2 ** s["step"] for s in states])
        torch._foreach_sqrt_(denominators)
        torch._foreach_add_(denominators, group["eps"])
        updates = torch._foreach_div(exp_avgs, [1 - beta1 ** s["step"] for s in states])
        torch._foreach_div_(updates, denominators)
        return updates

    def _muon_updates(self, params: List[Tensor], group: dict, rows: bool = False) -> List[Tensor]:
        """Muon's updates, or with ``rows`` NorMuon's."""
        for p in params:
            if not self.state[p]:
                self.state[p]["momentum_buffer"] = torch.zeros_like(p)
                if rows:
                    self.state[p]["row_second_moment"] = p.new_zeros(p.shape[0], 1)
        if not params:
            return []
        grads, buffers = [p.grad for p in params], [self.state[p]["momentum_buffer"] for p in params]
        torch._foreach_lerp_(buffers, grads, 1 - group["momentum"])
        nesterov = torch._foreach_lerp(grads, buffers, group["momentum"])
        if not rows:
            orthogonalize = partial(orthogonalized, method=group["orthogonalization"])
            return _batched(orthogonalize, nesterov, group["roles"])
        normalize = partial(row_normalized, beta=group["row_beta"], method=group["orthogonalization"])
        seconds = [self.state[p]["row_second_moment"] for p in params]
        return _batched(normalize, nesterov, group["roles"], states=seconds)

    def _soap_updates(self, params: List[Tensor], group: dict) -> List[Optional[Tensor]]:
        """SOAP's updates, None for the first gradient of a matrix. The state of a matrix holds one matrix per role:
        ``exp_avg`` in the coordinates of the weight, ``exp_avg_sq`` in the eigenbases, and the Gram matrices and
        eigenbases of the rotated sides (``gram_left``, ``basis_left``, ``gram_right``, ``basis_right``)."""
        gram_beta = group["betas"][1] if group["shampoo_beta"] is None else group["shampoo_beta"]
        grads = [roles_of(p.grad, group["roles"]) for p in params]
        states = [self.state[p] for p in params]
        updates = [None] * len(params)
        started = []
        for i, (state, grad) in enumerate(zip(states, grads)):
            if state:
                started.append(i)
            else:
                self._start_soap(state, grad, group["one_sided"], gram_beta)
        # one batch per shape, step and rotated sides
        keys = [(grads[i].shape, states[i]["step"], *(f"basis_{s}" in states[i] for s in _SIDES)) for i in started]
        for batch in _groups(keys):
            indices = [started[k] for k in batch]
            update = self._soap_step([states[i] for i in indices], [grads[i] for i in indices], group, gram_beta)
            for i, matrix in zip(indices, update.split(group["roles"])):
                updates[i] = matrix.reshape(params[i].shape)
        return updates

    @staticmethod
    def _start_soap(state: dict, grad: Tensor, one_sided: bool, gram_beta: float) -> None:
        rows, columns = grad.shape[-2:]
        state["step"] = 0
        state["exp_avg"] = torch.zeros_like(grad)
        state["exp_avg_sq"] = torch.zeros_like(grad)
        for side, smaller in (("left", rows < columns), ("right", columns <= rows)):
            if smaller or not one_sided:
                state[f"gram_{side}"] = (1 - gram_beta) * _SIDES[side](grad)
                state[f"basis_{side}"] = eigenbasis(state[f"gram_{side}"])

    @staticmethod
    def _soap_step(states: List[dict], grads: List[Tensor], group: dict, gram_beta: float) -> Tensor:
        """The update of matrices of one shape, step and rotated sides, as one batch of their roles."""
        beta1, beta2 = group["betas"]
        roles = group["roles"]
        for state in states:
            state["step"] += 1
        step = states[0]["step"]
        grad = torch.cat(grads)
        bases = {
            side: torch.cat([s[f"basis_{side}"] for s in states]) if f"basis_{side}" in states[0] else None
            for side in _SIDES
        }
        exp_avgs, exp_avg_sqs = [s["exp_avg"] for s in states], [s["exp_avg_sq"] for s in states]
        rotated = _rotate(grad, bases["left"], bases["right"]).split(roles)
        torch._foreach_lerp_(exp_avgs, grads, 1 - beta1)
        torch._foreach_mul_(exp_avg_sqs, beta2)
        torch._foreach_addcmul_(exp_avg_sqs, rotated, rotated, value=1 - beta2)
        second = torch.cat(exp_avg_sqs)
        denominator = (second / (1 - beta2**step)).sqrt_().add_(group["eps"])
        update = _rotate(torch.cat(exp_avgs), bases["left"], bases["right"]) / (1 - beta1**step) / denominator
        update = _rotate_back(update, bases["left"], bases["right"])

        # the gradient joins the preconditioner after its step, as in the reference implementation
        refresh = step % group["precondition_frequency"] == 0
        for (side, gram_of), dim in zip(_SIDES.items(), (0, 1)):
            if bases[side] is None:
                continue
            grams = [s[f"gram_{side}"] for s in states]
            torch._foreach_lerp_(grams, gram_of(grad).split(roles), 1 - gram_beta)
            if refresh:
                basis, second = refreshed_basis(torch.cat(grams), bases[side], second, dim)
                torch._foreach_copy_([s[f"basis_{side}"] for s in states], basis.split(roles))
        if refresh:
            torch._foreach_copy_(exp_avg_sqs, second.split(roles))
        return update


def refuse_weight_norm(model: nn.Module) -> None:
    """Raise if the equivariant blocks of ``model`` have weight-normalized layers (``weight_norm=true``).

    Weight normalization already sets the update size of these matrices: it re-projects every one of them
    after each optimizer step. Its layers are also no ``nn.Linear``, so ``block_param_groups`` would leave
    them out of the block group without notice, or find no block matrix at all.
    """
    normalized = [
        f"{block_name}.{name}"
        for block_name, block in model.named_modules()
        if isinstance(block, tuple(_BLOCKS))
        for name, layer in block.named_modules()
        if isinstance(layer, NormalizedDense)
    ]
    if normalized:
        more = f" and {len(normalized) - 3} more" if len(normalized) > 3 else ""
        raise ValueError(
            "weight_norm=true cannot be combined with task/optimizer=block_adam (task.block_scope): both set the "
            "update size of the weight matrices of the equivariant blocks. Use one of them, e.g. "
            f"globals.weight_norm=false. Weight-normalized block layers: {', '.join(normalized[:3])}{more}"
        )


def block_param_groups(
    params: Iterable[Tensor], model: nn.Module, scope: str = "all", slices: bool = False
) -> List[dict]:
    """Split ``params`` into the weight matrices of the equivariant blocks of ``model`` (``block=True``) and the rest.

    ``scope="all"``: every linear layer of the PaiNN interaction and mixing blocks and of the gated equivariant
    blocks of the head. ``scope="vector"``: only the ones acting on vector features (``mu_channel_mix``,
    ``mix_vectors``), which mix the channels the same way for all 3 components: the l = 1 irrep blocks.
    Embeddings, the filter network, biases and normalization gains are never blocks, and the rest never takes
    weight decay: ``weight_decay`` of the optimizer decays the block matrices only.

    ``slices=True`` gives the fused layers (``_ROLES``: the layers whose output the forward pass splits into 2 or 3
    roles) the ``roles`` of their group, so that ``BlockAdamW`` steps each role as a block of its own. The block
    matrices then come in one group per number of roles, in increasing order. The last layer of the gated blocks'
    scalar net splits its output unevenly (scalars and gates), and stays one block.
    """
    if scope not in ("all", "vector"):
        raise ValueError(f"unknown block scope {scope!r}")
    refuse_weight_norm(model)
    roles = {}  # id of a block matrix -> number of roles of its layer
    for module in model.modules():
        for block_cls, vector_layer in _BLOCKS.items():
            if isinstance(module, block_cls):
                for name, layer in module.named_modules():
                    if isinstance(layer, nn.Linear) and (scope == "all" or name == vector_layer):
                        roles[id(layer.weight)] = _ROLES.get(name, 1) if slices else 1
    params = list(params)
    matrices = [p for p in params if id(p) in roles]
    if not matrices:
        raise ValueError(f"no weight matrices of equivariant blocks in the parameters for block scope {scope!r}")
    rest = [p for p in params if id(p) not in roles]
    blocks = [
        {"params": [p for p in matrices if roles[id(p)] == n], "block": True, "roles": n}
        for n in sorted({roles[id(p)] for p in matrices})
    ]
    return blocks + ([{"params": rest, "weight_decay": 0.0}] if rest else [])
