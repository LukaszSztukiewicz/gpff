"""D2: held-out conditionals and fixed-geometry any-order decoding."""
import torch
from schnetpack import properties as P
from gpff import keys, vocab
from gpff.analysis.absorbing_diagnostics import snapshot


def conditional_metrics(probabilities, true_z, selected, generator, bins=10):
    p = probabilities[selected]
    truth = vocab.z_to_index(true_z[selected])
    if not len(p):
        return {"n_atoms": 0}
    if not bool(torch.isfinite(p).all()) or bool((p.sum(-1) <= 0).any()):
        return {"n_atoms": len(p), "nonfinite_predictions": int((~torch.isfinite(p).all(-1)).sum()), "failed": True}
    p = p / p.sum(-1, keepdim=True)
    draw = torch.multinomial(p, 1, generator=generator).squeeze(-1)
    argmax = p.argmax(-1)
    target = torch.nn.functional.one_hot(truth, vocab.n_tokens()).to(p.dtype)
    result = {"n_atoms": len(p), "nll": float(-p.gather(1, truth[:, None]).clamp_min(1e-30).log().mean()),
              "brier": float((p - target).square().sum(-1).mean()),
              "entropy": float(-(p * p.clamp_min(1e-30).log()).sum(-1).mean()),
              "sample_argmax_disagreement": float((draw != argmax).float().mean()),
              "expected_counts": p.sum(0).tolist(), "true_counts": target.sum(0).tolist(), "elements": {}}
    for k, z in enumerate(vocab.TOKEN_Z):
        if z in (vocab.MASK_Z, vocab.ABSENT_Z):
            continue
        at = truth == k
        calibration = []
        for b in range(bins):
            sel = ((p[:, k] * bins).long().clamp(max=bins-1) == b)
            calibration.append({"n": int(sel.sum()), "mean_probability": float(p[sel, k].mean()) if bool(sel.any()) else None,
                                "frequency": float(at[sel].float().mean()) if bool(sel.any()) else None})
        result["elements"][str(z)] = {"n_true": int(at.sum()),
            "argmax_recall": float((argmax[at] == k).float().mean()) if bool(at.any()) else None,
            "sample_recall": float((draw[at] == k).float().mean()) if bool(at.any()) else None,
            "reliability": calibration}
    return result


@torch.no_grad()
def fixed_geometry_decode(calculator, batch, true_z, generator, mode="generated", parallel=False):
    """Return completed tokens; clean context is a reconstruction diagnostic.

    Positions are fixed, the initial visible context is shared, and a random
    independent reveal order is drawn per molecule. Generated context quality
    must be assessed distributionally rather than only against original labels.
    """
    if mode not in ("generated", "clean"):
        raise ValueError("Unknown context mode")
    work = snapshot(batch)
    original_positions = work[P.R].clone()
    pending = work[P.Z] == vocab.MASK_Z
    order = torch.rand(len(pending), device=pending.device, generator=generator)
    output = work[P.Z].clone()
    calls = 0
    while bool(pending.any()):
        pred = calculator.calculate(work, forces=True, x0=True)
        calls += 1
        selected = pending.clone()
        if not parallel:
            selected.zero_()
            for m in torch.unique(work[P.idx_m][pending]):
                slots = torch.where(pending & (work[P.idx_m] == m))[0]
                selected[slots[order[slots].argmin()]] = True
        p = pred["type_probabilities"][selected]
        if not bool(torch.isfinite(p).all()):
            raise ValueError("Nonfinite fixed-geometry conditional")
        indices = torch.multinomial(p, 1, generator=generator).squeeze(-1)
        output[selected] = vocab.index_to_z(indices).to(output.dtype)
        work[P.Z][selected] = true_z[selected] if mode == "clean" else output[selected]
        pending &= ~selected
    torch.testing.assert_close(work[P.R], original_positions)
    return output, calls
