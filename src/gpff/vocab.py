"""
The atom type vocabulary shared by the discrete type models.

One fixed, ordered vocabulary

    {ABSENT} u {H, B, C, N, O, F, Al, Si, P, S, Cl, As, Se, Br, I, Hg, Bi} u {MASK}

for every dataset, with a per-dataset mask over the tokens a model may emit. The
order is **frozen**: a token's index is part of every checkpoint, so inserting
one invalidates them all. It is a superset of QM9 (H, C, N, O, F), the
CrossDocked2020 ligands (C, N, O, F, P, S, Cl, Br, I) and GEOM-Drugs, so a model
trained on one can be fine-tuned on another without reindexing.

Two tokens are not elements:

    ABSENT  no atom in this slot. A normal class of the vocabulary, which is
            what lets a model generate the number of atoms rather than being
            given it. Carried as Z = 0, which ASE already reads as its dummy
            element X.
    MASK    the absorbing state of the forward process: the type is not known
            yet. Carried as Z = 99, an atomic number no dataset uses, so PaiNN's
            nn.Embedding(100, ...) gives it a row of its own and nothing else in
            the model has to change.

A model must never emit MASK (the reverse process carries unmasked tokens over
unchanged and only ever replaces MASK), and must not emit an element its dataset
does not contain: an untrained logit is worse than an untrained embedding,
because it can be sampled. ``disallowed_mask`` builds the mask that enforces
both.
"""

from typing import Dict, Iterable, Sequence, Tuple

import torch


__all__ = [
    "ABSENT",
    "MASK",
    "TOKENS",
    "TOKEN_Z",
    "ABSENT_INDEX",
    "MASK_INDEX",
    "MASK_Z",
    "ABSENT_Z",
    "n_tokens",
    "index_of_token",
    "z_to_index",
    "index_to_z",
    "disallowed_mask",
    "QM9_ELEMENTS",
]

ABSENT = "ABSENT"
MASK = "MASK"

# frozen order; append new elements before MASK only if no checkpoint matters
TOKENS: Tuple[str, ...] = (
    ABSENT,
    "H", "B", "C", "N", "O", "F", "Al", "Si", "P",
    "S", "Cl", "As", "Se", "Br", "I", "Hg", "Bi",
    MASK,
)

ABSENT_Z = 0
MASK_Z = 99

# atomic number carried in properties.Z for each token
TOKEN_Z: Tuple[int, ...] = (
    ABSENT_Z,
    1, 5, 6, 7, 8, 9, 13, 14, 15,
    16, 17, 33, 34, 35, 53, 80, 83,
    MASK_Z,
)

ABSENT_INDEX = TOKENS.index(ABSENT)
MASK_INDEX = TOKENS.index(MASK)

QM9_ELEMENTS: Tuple[str, ...] = ("H", "C", "N", "O", "F")

_INDEX_OF_TOKEN: Dict[str, int] = {token: i for i, token in enumerate(TOKENS)}


def n_tokens() -> int:
    """The size of the vocabulary, i.e. the width of a type head."""
    return len(TOKENS)


def index_of_token(token: str) -> int:
    """
    The vocabulary index of a token, by name ("C") or by atomic number ("6").
    """
    if token in _INDEX_OF_TOKEN:
        return _INDEX_OF_TOKEN[token]
    try:
        z = int(token)
    except (TypeError, ValueError):
        raise KeyError(
            f"Unknown token {token!r}; the vocabulary is {', '.join(TOKENS)}."
        ) from None
    if z not in TOKEN_Z:
        raise KeyError(f"No token carries atomic number {z}.")
    return TOKEN_Z.index(z)


def z_to_index(z: torch.Tensor) -> torch.Tensor:
    """
    Map atomic numbers (as carried in ``properties.Z``) to vocabulary indices.

    Raises ValueError on an atomic number outside the vocabulary, rather than
    silently mapping it to ABSENT: that would turn an unsupported element into
    a missing atom.
    """
    table = torch.full((MASK_Z + 1,), -1, dtype=torch.long, device=z.device)
    table[torch.tensor(TOKEN_Z, device=z.device)] = torch.arange(
        len(TOKEN_Z), device=z.device
    )
    if (z < 0).any() or (z > MASK_Z).any():
        raise ValueError(f"Atomic numbers out of range: {z.min()}..{z.max()}.")
    indices = table[z]
    if (indices < 0).any():
        missing = sorted({int(v) for v in z[indices < 0].unique()})
        raise ValueError(f"Atomic numbers outside the vocabulary: {missing}.")
    return indices


def index_to_z(indices: torch.Tensor) -> torch.Tensor:
    """Map vocabulary indices back to the atomic numbers of ``properties.Z``."""
    table = torch.tensor(TOKEN_Z, dtype=torch.long, device=indices.device)
    return table[indices]


def disallowed_mask(
    allowed: Sequence[str] = QM9_ELEMENTS,
    allow_absent: bool = False,
) -> torch.Tensor:
    """
    A boolean mask over the vocabulary, True for the tokens a model must not emit.

    MASK is always disallowed. ABSENT only with ``allow_absent``, which a model
    generating the number of atoms sets.

    Args:
        allowed: the element tokens of the dataset, by name or atomic number.
        allow_absent: whether ABSENT is a legal prediction.
    """
    mask = torch.ones(n_tokens(), dtype=torch.bool)
    for token in allowed:
        index = index_of_token(str(token))
        if index in (ABSENT_INDEX, MASK_INDEX):
            raise ValueError(
                f"{TOKENS[index]} is not an element; pass allow_absent for ABSENT."
            )
        mask[index] = False
    if allow_absent:
        mask[ABSENT_INDEX] = False
    return mask
