"""Simultaneous random ReMDM transitions (Wang et al., equation 6).

Here r=1-alpha_t, s=1-alpha_s and b is visible-to-mask probability.
The confidence ablation is state dependent and has no exact marginal claim.
"""
import math


def transition_probabilities(r: float, s: float, strength: float = 0.0):
    """Return (masked-to-visible, visible-to-mask) with a rescaled bound."""
    if not all(math.isfinite(v) for v in (r, s, strength)):
        raise ValueError("Rates and strength must be finite")
    if not 0 <= s <= r <= 1 or not 0 <= strength <= 1:
        raise ValueError("Require 0 <= next rate <= current rate <= 1 and strength in [0,1]")
    if r == 0:
        return 0.0, 0.0
    if s == 0:
        return 1.0, 0.0
    b = strength * (min(1.0, s / (1 - r)) if r < 1 else 1.0)
    a = (r - s + (1 - r) * b) / r
    return min(1.0, max(0.0, a)), b
