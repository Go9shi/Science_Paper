"""Causal volatility sensitivity; alpha=0 is exactly the fixed penalty."""
import math


def penalty_coefficient(mode, lambda_0, sigma, sigma_ref, *, alpha=1., ratio_cap=5.,
                        ratio_floor=1e-6, lambda_min=0., lambda_max=None):
    numbers = (lambda_0, sigma, sigma_ref, alpha, ratio_cap, ratio_floor, lambda_min)
    if not all(math.isfinite(x) for x in numbers):
        raise ValueError("Penalty parameters must be finite")
    if (mode not in {"none", "fixed", "adaptive"} or lambda_0 < 0 or sigma < 0 or sigma_ref <= 0
            or ratio_cap < 1 or not 0 < ratio_floor <= 1 or not 0 <= lambda_min <= lambda_0):
        raise ValueError("Invalid penalty mode/bounds/reference")
    if lambda_max is not None and (not math.isfinite(lambda_max) or lambda_max < lambda_0):
        raise ValueError("lambda_max must be finite and at least lambda_0")
    if mode == "none":
        return 0.
    if mode == "fixed" or alpha == 0:
        return float(lambda_0)
    ratio = sigma / sigma_ref
    if lambda_0 == 0 or (ratio == 0 and alpha > 0):
        value = 0.
    elif alpha == 1:
        # Preserve the verified Stage 2 rule, including the sigma=0 case.
        value = lambda_0 * min(ratio, ratio_cap)
    else:
        safe_ratio = max(ratio, ratio_floor) if alpha < 0 else ratio
        exponent = min(alpha * math.log(safe_ratio), math.log(ratio_cap))
        value = lambda_0 * math.exp(max(-745., exponent))
    result = float(max(lambda_min, min(value, lambda_max if lambda_max is not None else math.inf)))
    if not math.isfinite(result):
        raise FloatingPointError("Penalty coefficient overflow")
    return result
