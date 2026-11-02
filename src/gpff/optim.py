"""AdamW with a controlled update size for the weight matrices of equivariant blocks.

An equivariant layer mixes channels with one matrix that is shared over the 2l + 1
components of a feature (in PaiNN: the 3 components of a vector). Adam rescales every
entry on its own, so one learning rate still gives these matrices very different step
sizes. ``BlockAdamW`` rescales the update of each such matrix ("block") so that the
learning rate is its size, or replaces it by Muon's or SOAP's update.

Manolache & Niepert, "Equivariance Breaks the Learning Rate" (NeurIPS 2026 workshop).
Vyas et al., "SOAP: Improving and Stabilizing Shampoo using Adam" (ICLR 2025).

A network has a few dozen of these matrices, and each of them is small (256 x 256), so a step is
bound by the launch of GPU kernels, not by their arithmetic. The step therefore never loops over
parameters: it works on lists of tensors (``torch._foreach_*``) and runs the matrix operations of all
matrices of one shape, and of all roles of a fused layer, as one batched operation.
"""
import math
from typing import Iterable, List, Optional

import torch
from schnetpack.nn import GatedEquivariantBlock
from schnetpack.representation.painn import PaiNNInteraction, PaiNNMixing
from torch import Tensor, nn
from torch.optim import Optimizer

__all__ = ["BlockAdamW", "block_param_groups"]

# the equivariant blocks, with the name of their layer acting on vector features (None: no such layer)
_BLOCKS = {PaiNNInteraction: None, PaiNNMixing: "mu_channel_mix", GatedEquivariantBlock: "mix_vectors"}
# the fused layers of the blocks, by name in the block: one weight whose rows are the outputs of this many layers of
# the same size computed as one, which the forward pass splits into the roles of the layer
_ROLES = {"interatomic_context_net.1": 3, "intraatomic_context_net.1": 3, "mu_channel_mix": 2, "mix_vectors": 2}
# the RMS of an AdamW update entry, which Muon's update is scaled to
_ADAMW_RMS = 0.2


# The functions on matrices take one matrix or a batch of them (the last two dimensions are a matrix).


def rms_norm(update: Tensor) -> Tensor:
    """The RMS of each matrix: a scalar for one matrix, one number per matrix for a batch."""
    return update.pow(2).mean(dim=(-2, -1)).sqrt()


def spectral_norm(update: Tensor) -> Tensor:
    """Schatten-16 norm: an upper bound of the largest singular value, at most rank**(1/16) times it
    (exact for rank one). Two squarings of the Gram matrix replace an SVD, which is too slow for wide matrices."""
    frobenius = torch.linalg.vector_norm(update, dim=(-2, -1))
    # singular values of at most one cannot overflow in the 16th power
    unit = update / frobenius.clamp_min(torch.finfo(update.dtype).tiny)[..., None, None]
    gram = unit.mT @ unit if unit.shape[-2] >= unit.shape[-1] else unit @ unit.mT
    gram = gram @ gram
    gram = gram @ gram
    return frobenius * torch.linalg.vector_norm(gram, dim=(-2, -1)) ** (1 / 8)


def newton_schulz(matrix: Tensor, steps: int = 5) -> Tensor:
    """Muon's orthogonalization: the matrix with the singular vectors of ``matrix`` and singular values near one."""
    a, b, c = 3.4445, -4.7750, 2.0315
    single = matrix.ndim == 2
    x = matrix[None] if single else matrix
    x = x / torch.linalg.vector_norm(x, dim=(-2, -1), keepdim=True).clamp_min(1e-7)
    tall = x.shape[-2] > x.shape[-1]
    if tall:
        x = x.mT
    for _ in range(steps):
        gram = x @ x.mT
        # b * gram + c * gram @ gram and a * x + (that) @ x, each as one fused matrix product
        x = torch.baddbmm(x, torch.baddbmm(gram, gram, gram, beta=b, alpha=c), x, beta=a)
    x = x.mT if tall else x
    return x[0] if single else x


_NORMS = {"rms": rms_norm, "spectral": spectral_norm}


def orthogonalized(matrix: Tensor) -> Tensor:
    """Muon's update of a gradient: Newton-Schulz, scaled to the RMS of an AdamW update."""
    return newton_schulz(matrix) * (_ADAMW_RMS * math.sqrt(max(matrix.shape[-2:])))


def normalized(update: Tensor, norm: str) -> Tensor:
    """``update`` divided by its RMS or spectral norm (zero stays zero)."""
    return update / _NORMS[norm](update).clamp_min(torch.finfo(update.dtype).tiny)[..., None, None]


def accumulate_grams(state: dict, grad: Tensor, beta: float) -> None:
    """Average ``grad @ grad.T`` and ``grad.T @ grad`` into the Gram matrices of the state of a SOAP matrix (those
    of the sides it preconditions)."""
    if "gram_left" in state:
        state["gram_left"].lerp_(grad @ grad.T, 1 - beta)
    if "gram_right" in state:
        state["gram_right"].lerp_(grad.T @ grad, 1 - beta)


def soap_sides(rows: int, columns: int, one_sided: bool):
    """Whether SOAP rotates the left and the right side of a ``rows`` x ``columns`` matrix: both, or with
    ``one_sided`` only the smaller one (the right one if they are equal)."""
    if not one_sided:
        return True, True
    return rows < columns, columns <= rows


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


# The updates of one group of parameters, on the lists of its parameters, gradients and state tensors, in place.
# ``lr`` and the bias corrections ``bc1``, ``bc2`` are floats, or 0-dim tensors when the functions are compiled
# (a new float would compile them again at every step).


def _by_shape(tensors: List[Tensor]) -> List[List[int]]:
    """The indices of ``tensors`` grouped by shape, in order of first appearance."""
    groups = {}
    for i, tensor in enumerate(tensors):
        key = tuple(tensor.shape)
        if key not in groups:
            groups[key] = []
        groups[key].append(i)
    return list(groups.values())


def _stacked(function, matrices: List[Tensor], roles: int) -> List[Tensor]:
    """``function`` of each of the ``roles`` equal slices of the rows of every matrix on its own: one batched call per
    shape of slice. Returns the results with the shapes of ``matrices``."""
    results = [None] * len(matrices)
    for indices in _by_shape(matrices):
        rows, columns = matrices[indices[0]].shape
        batch = torch.cat([matrices[i].reshape(-1, rows // roles, columns) for i in indices])
        for i, result in zip(indices, function(batch).reshape(len(indices), rows, columns).unbind(0)):
            results[i] = result
    return results


def _apply(params: List[Tensor], updates: List[Tensor], lr, weight_decay: float) -> None:
    """Decay the weights, then take the step ``lr * update`` (``updates`` is overwritten)."""
    if weight_decay:
        torch._foreach_mul_(params, 1 - lr * weight_decay)
    torch._foreach_mul_(updates, lr)
    torch._foreach_sub_(params, updates)


def _update_adam(
    params, grads, exp_avgs, exp_avg_sqs, lr, bc1, bc2, *, betas, eps, weight_decay, normalize, roles
) -> None:
    beta1, beta2 = betas
    torch._foreach_lerp_(exp_avgs, grads, 1 - beta1)
    torch._foreach_mul_(exp_avg_sqs, beta2)
    torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1 - beta2)
    denominators = torch._foreach_div(exp_avg_sqs, bc2)
    torch._foreach_sqrt_(denominators)
    torch._foreach_add_(denominators, eps)
    updates = torch._foreach_div(exp_avgs, bc1)
    torch._foreach_div_(updates, denominators)
    if normalize is not None:
        updates = _stacked(lambda batch: normalized(batch, normalize), updates, roles)
    _apply(params, updates, lr, weight_decay)


def _update_muon(params, grads, buffers, lr, *, momentum, weight_decay, roles) -> None:
    torch._foreach_lerp_(buffers, grads, 1 - momentum)
    updates = _stacked(orthogonalized, torch._foreach_lerp(grads, buffers, momentum), roles)
    _apply(params, updates, lr, weight_decay)


def _rotate(x: Tensor, left: Optional[Tensor], right: Optional[Tensor]) -> Tensor:
    """``left.T @ x @ right`` into the bases of SOAP; a side without a basis is not rotated."""
    if left is not None:
        x = left.mT @ x
    if right is not None:
        x = x @ right
    return x


def _rotate_back(x: Tensor, left: Optional[Tensor], right: Optional[Tensor]) -> Tensor:
    """``left @ x @ right.T`` out of the bases of SOAP."""
    if left is not None:
        x = left @ x
    if right is not None:
        x = x @ right.mT
    return x


def _update_soap(
    params, grads, exp_avgs, exp_avg_sqs, gram_lefts, gram_rights, basis_lefts, basis_rights, lr, bc1, bc2,
    *, betas, eps, gram_beta, weight_decay, refresh, one_sided,
) -> None:
    # the lists of a side that is not rotated (``one_sided``) hold None
    beta1, beta2 = betas
    torch._foreach_lerp_(exp_avgs, grads, 1 - beta1)
    updates = [None] * len(params)
    for indices in _by_shape(params):
        rows, columns = params[indices[0]].shape
        use_left, use_right = soap_sides(rows, columns, one_sided)
        grad, first = torch.stack([grads[i] for i in indices]), torch.stack([exp_avgs[i] for i in indices])
        left = torch.stack([basis_lefts[i] for i in indices]) if use_left else None
        right = torch.stack([basis_rights[i] for i in indices]) if use_right else None
        second = [exp_avg_sqs[i] for i in indices]
        rotated = _rotate(grad, left, right).unbind(0)
        torch._foreach_mul_(second, beta2)
        torch._foreach_addcmul_(second, rotated, rotated, value=1 - beta2)
        denominator = (torch.stack(second) / bc2).sqrt_().add_(eps)
        update = _rotate_back(_rotate(first, left, right) / bc1 / denominator, left, right)
        # the gradient joins the preconditioner after its step: it did not rotate itself
        grams_left = [gram_lefts[i] for i in indices] if use_left else None
        grams_right = [gram_rights[i] for i in indices] if use_right else None
        if use_left:
            torch._foreach_lerp_(grams_left, (grad @ grad.mT).unbind(0), 1 - gram_beta)
        if use_right:
            torch._foreach_lerp_(grams_right, (grad.mT @ grad).unbind(0), 1 - gram_beta)
        if refresh:
            sorted_second = torch.stack(second)
            if use_left:
                new_left, sorted_second = refreshed_basis(torch.stack(grams_left), left, sorted_second, 0)
                torch._foreach_copy_([basis_lefts[i] for i in indices], new_left.unbind(0))
            if use_right:
                new_right, sorted_second = refreshed_basis(torch.stack(grams_right), right, sorted_second, 1)
                torch._foreach_copy_([basis_rights[i] for i in indices], new_right.unbind(0))
            torch._foreach_copy_(second, sorted_second.unbind(0))
        for i, step in zip(indices, update.unbind(0)):
            updates[i] = step
    _apply(params, updates, lr, weight_decay)


_UPDATES = {"adam": _update_adam, "muon": _update_muon, "soap": _update_soap}
_COMPILED = {}


def _compiled(update):
    if update not in _COMPILED:
        import torch._dynamo

        # one version of the update per group of parameters, and per refresh of SOAP's bases
        config = torch._dynamo.config
        limit = "recompile_limit" if hasattr(config, "recompile_limit") else "cache_size_limit"
        setattr(config, limit, max(getattr(config, limit), 64))
        _COMPILED[update] = torch.compile(update, dynamic=False)
    return _COMPILED[update]


class BlockAdamW(Optimizer):
    """AdamW; the matrices of parameter groups with ``block=True`` (see ``block_param_groups``) take
    ``block_update`` steps instead:

    * ``"adam"``: AdamW's.
    * ``"rms"``, ``"spectral"``: AdamW's update divided by its RMS or spectral norm, so that the step has
      exactly this norm times ``lr`` and the direction of AdamW's. Single learning rate for all blocks;
      zero updates stay zero.
    * ``"muon"``: Muon's update, Nesterov ``momentum`` and Newton-Schulz orthogonalization (no Adam moments),
      scaled to the RMS of an AdamW update, so ``lr`` is comparable to AdamW's.
    * ``"soap"``: Adam in the eigenbasis of Shampoo's preconditioner. Every matrix keeps the averages
      (``shampoo_beta``, default ``betas[1]``) of ``G G^T`` and ``G^T G`` over its gradients ``G``, and the Adam
      moments are those of ``G`` rotated into their eigenvectors, which are refreshed every
      ``precondition_frequency`` steps. The rotation keeps the size of the update, so ``lr`` is AdamW's.
      The first gradient only starts the preconditioner and takes no step, as in the reference implementation.
      Eigenvectors cost cubic time in the sides of a matrix: the 3F x F matrices of the fused layers have a
      side of 3F, and a larger ``precondition_frequency`` makes that cheaper. Fused layers are not split into
      ``roles``.

    ``weight_decay`` is decoupled, as in AdamW: a step shrinks the weights by ``lr * weight_decay`` before the
    update, whichever update it is. It is off by default; the groups of ``block_param_groups`` decay the block
    matrices only.

    A block group may set ``roles`` (default 1) for a fused layer: its weight stacks that many equal matrices,
    one per role of the layer's output. ``"rms"``, ``"spectral"`` and ``"muon"`` then take the step of each of
    them on its own, as if the roles were separate layers, instead of one step for the stack: the size of a
    role's step does not depend on the gradients of the other roles.

    The step works on lists of tensors and batches the matrices of one shape (see the top of the module). The
    state of every parameter is its own tensors, as in torch's optimizers. All parameters of a group have the
    same dtype and device.

    ``compiled=True`` compiles the update of every group with ``torch.compile``: the first steps take longer, and
    the many small element-wise operations around the matrix products become a few kernels.

    ``one_sided=True`` makes ``"soap"`` rotate only the smaller side of every matrix (the right one if they are
    equal), as in the one-sided variant of the SOAP paper: the other side keeps the coordinates of the weight. A 3F x F
    matrix of a fused layer then has a Gram matrix, eigenbases and refreshes of size F instead of 3F, which is what
    the refresh costs (cubic in the side). Like ``compiled``, it is not in the state dict: state saved with the
    other setting is rejected.
    """

    # class-level defaults, so optimizers pickled before the options existed still load
    compiled: bool = False
    one_sided: bool = False

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
        compiled: bool = False,
        one_sided: bool = False,
    ):
        if block_update not in ("adam", "muon", "soap", *_NORMS):
            raise ValueError(f"unknown block_update {block_update!r}")
        if precondition_frequency < 1:
            raise ValueError("precondition_frequency has to be at least 1")
        if shampoo_beta is not None and not 0 <= shampoo_beta < 1:
            raise ValueError("shampoo_beta has to be in [0, 1)")
        defaults = dict(
            lr=lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay,
            block=False, block_update=block_update, momentum=momentum, roles=1,
            precondition_frequency=precondition_frequency, shampoo_beta=shampoo_beta,
        )
        self.compiled = compiled
        self.one_sided = one_sided
        super().__init__(params, defaults)

    def add_param_group(self, param_group):
        super().add_param_group(param_group)
        group = self.param_groups[-1]
        if group["block"] and any(p.ndim != 2 for p in group["params"]):
            raise ValueError("the parameters of a block group have to be weight matrices")
        if group["roles"] < 1 or (group["block"] and any(p.shape[0] % group["roles"] for p in group["params"])):
            raise ValueError("the rows of a block matrix have to split into the roles of its group")
        if group["block"] and group["block_update"] == "soap" and group["roles"] != 1:
            raise ValueError("soap preconditions whole matrices: it does not split fused layers into roles")
        if len({(p.dtype, p.device) for p in group["params"]}) > 1:
            raise ValueError("the parameters of a group have to share their dtype and device")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            states = [self.state[p] for p in params]
            mode = group["block_update"] if group["block"] else "adam"
            if mode == "muon":
                self._step_muon(group, params, states)
            elif mode == "soap":
                self._step_soap(group, params, states)
            else:
                self._step_adam(group, params, states, mode)
        return loss

    def _update(self, name: str):
        return _compiled(_UPDATES[name]) if self.compiled else _UPDATES[name]

    def _scalars(self, like: Tensor, lr: float, bc1: float = 1.0, bc2: float = 1.0):
        """The learning rate and the bias corrections: floats, or when compiling 0-dim tensors on the device of
        ``like``, written by kernels (a copy from the host would wait for the device)."""
        if not self.compiled:
            return lr, bc1, bc2
        buffers = self.__dict__.setdefault("_buffers", {})
        if like.device not in buffers:
            buffers[like.device] = torch.empty(3, dtype=like.dtype, device=like.device)
        buffer = buffers[like.device]
        for i, value in enumerate((lr, bc1, bc2)):
            buffer[i].fill_(value)
        return buffer[0], buffer[1], buffer[2]

    @staticmethod
    def _by_step(states: List[dict]) -> List[List[int]]:
        """The indices of ``states`` grouped by their step (all the same, unless a gradient was ever missing)."""
        steps = {}
        for i, state in enumerate(states):
            steps.setdefault(state["step"], []).append(i)
        return list(steps.values())

    def _step_adam(self, group: dict, params: List[Tensor], states: List[dict], mode: str) -> None:
        for p, state in zip(params, states):
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
        beta1, beta2 = group["betas"]
        for indices in self._by_step(states):
            part, part_states = [params[i] for i in indices], [states[i] for i in indices]
            step = part_states[0]["step"] + 1
            for state in part_states:
                state["step"] = step
            self._update("adam")(
                part,
                [p.grad for p in part],
                [state["exp_avg"] for state in part_states],
                [state["exp_avg_sq"] for state in part_states],
                *self._scalars(part[0], group["lr"], 1 - beta1**step, 1 - beta2**step),
                betas=(beta1, beta2),
                eps=group["eps"],
                weight_decay=group["weight_decay"],
                normalize=mode if mode in _NORMS else None,
                roles=group["roles"],
            )

    def _step_muon(self, group: dict, params: List[Tensor], states: List[dict]) -> None:
        for p, state in zip(params, states):
            if not state:
                state["momentum_buffer"] = torch.zeros_like(p)
        self._update("muon")(
            params,
            [p.grad for p in params],
            [state["momentum_buffer"] for state in states],
            self._scalars(params[0], group["lr"])[0],
            momentum=group["momentum"],
            weight_decay=group["weight_decay"],
            roles=group["roles"],
        )

    def _step_soap(self, group: dict, params: List[Tensor], states: List[dict]) -> None:
        lr, (beta1, beta2) = group["lr"], group["betas"]
        gram_beta = beta2 if group["shampoo_beta"] is None else group["shampoo_beta"]
        is_new = [not state for state in states]
        for p, state, new in zip(params, states, is_new):
            if new:
                self._start_soap(p, state, gram_beta)
            else:
                sides = soap_sides(*p.shape, self.one_sided)
                if sides != ("basis_left" in state, "basis_right" in state):
                    raise ValueError(
                        f"the state of a SOAP matrix {tuple(p.shape)} is {'one' if self.one_sided else 'two'}-sided "
                        f"in this optimizer and was saved {'two' if self.one_sided else 'one'}-sided: "
                        "one_sided cannot change in the middle of a run"
                    )
        if any(is_new) and group["weight_decay"]:  # the first gradient takes no step, but the weights decay
            torch._foreach_mul_([p for p, new in zip(params, is_new) if new], 1 - lr * group["weight_decay"])
        started = [i for i, new in enumerate(is_new) if not new]
        for indices in self._by_step([states[i] for i in started]):
            part = [params[started[i]] for i in indices]
            part_states = [states[started[i]] for i in indices]
            step = part_states[0]["step"] + 1
            for state in part_states:
                state["step"] = step
            self._update("soap")(
                part,
                [p.grad for p in part],
                *([state.get(key) for state in part_states] for key in (
                    "exp_avg", "exp_avg_sq", "gram_left", "gram_right", "basis_left", "basis_right"
                )),
                *self._scalars(part[0], lr, 1 - beta1**step, 1 - beta2**step),
                betas=(beta1, beta2),
                eps=group["eps"],
                gram_beta=gram_beta,
                weight_decay=group["weight_decay"],
                refresh=step % group["precondition_frequency"] == 0,
                one_sided=self.one_sided,
            )

    def _start_soap(self, p: Tensor, state: dict, gram_beta: float) -> None:
        use_left, use_right = soap_sides(*p.shape, self.one_sided)
        state["step"] = 0
        state["exp_avg"] = torch.zeros_like(p)  # in the space of the weight
        state["exp_avg_sq"] = torch.zeros_like(p)  # in the eigenbasis
        if use_left:
            state["gram_left"] = p.new_zeros(p.shape[0], p.shape[0])
        if use_right:
            state["gram_right"] = p.new_zeros(p.shape[1], p.shape[1])
        accumulate_grams(state, p.grad, gram_beta)
        if use_left:
            state["basis_left"] = eigenbasis(state["gram_left"])
        if use_right:
            state["basis_right"] = eigenbasis(state["gram_right"])


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
