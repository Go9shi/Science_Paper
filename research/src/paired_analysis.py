"""Paired analysis of already evaluated paths. Execute numerical work in Colab."""
import numpy as np
import pandas as pd

from .financial_metrics import financial_metrics

METRICS = ("net_sharpe", "cagr", "maximum_drawdown", "portfolio_turnover_mean_daily", "total_commissions")
PAIRS = (("adaptive", "fixed"), ("fixed", "none"), ("adaptive", "none"))


def stationary_indices(length, block_length, rng):
    """Uniform starts, geometric block lengths, circular continuation."""
    if type(length) is not int or length < 2 or not 1 <= block_length <= length:
        raise ValueError("Invalid stationary bootstrap length/block")
    starts = rng.integers(0, length, size=length)
    restart = rng.random(length) < 1. / block_length
    result = np.empty(length, dtype=np.int64)
    result[0] = starts[0]
    for i in range(1, length):
        result[i] = starts[i] if restart[i] else (result[i - 1] + 1) % length
    return result


def _sharpe(values):
    """K=252, rf=0, ddof=1, matching the project's frozen metric convention."""
    if len(values) < 2:
        return np.full(values.shape[1:], np.nan)
    std = np.std(values, axis=0, ddof=1)
    return np.divide(np.mean(values, axis=0) * np.sqrt(252.), std,
                     out=np.full_like(std, np.nan), where=std > 1e-15)


def paired_analysis(records, *, seeds, sigma_ref, bootstrap):
    """records[(mode, seed)] are full ordered paths, never pooled day x seed.

    Bootstrap conditions on the finite trained-policy/seed set. It estimates
    time uncertainty; it does not resample seeds or reproduce a trading policy
    on a simulated market. Conditional regime Sharpe is descriptive, not CAGR.
    """
    modes = ("none", "fixed", "adaptive")
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or any(type(s) is not int for s in seeds):
        raise ValueError('Use at least two distinct preselected integer seeds')
    if set(records) != {(m, s) for m in modes for s in seeds} or not np.isfinite(sigma_ref) or sigma_ref <= 0:
        raise ValueError("Need all matched modes/seeds and a train-only sigma_ref")
    first = records[("none", seeds[0])]
    dates = first.date.tolist()
    if len(dates) < 2 or dates != sorted(set(dates)):
        raise ValueError("Dates must be unique and chronological")
    sigma = first.sigma_t.to_numpy(float)
    if not np.isfinite(sigma).all() or (sigma < 0).any():
        raise ValueError('Market sigma must be finite and nonnegative')
    per_run, regimes = [], []
    keys = [(m, s) for m in modes for s in seeds]
    for mode, seed in keys:
        frame = records[(mode, seed)]
        if frame.date.tolist() != dates or not np.array_equal(frame.sigma_t.to_numpy(float), sigma):
            raise ValueError("Pairing requires identical dates and causal market sigma")
        if not (frame.data_as_of < frame.date).all():
            raise ValueError("Feature as-of dates must precede decisions")
        metrics = financial_metrics(frame, 1000., annualization=252, risk_free_annual=0.)
        per_run.append({"mode": mode, "seed": seed, **metrics})
        for label, mask in (("high", sigma > sigma_ref), ("low", sigma <= sigma_ref)):
            part = frame.loc[mask]
            conditional = _sharpe(part.net_return.to_numpy(float).reshape(-1, 1))[0]
            regimes.append({"mode": mode, "seed": seed, "regime": label, "observations": len(part),
                "conditional_net_sharpe": float(conditional) if np.isfinite(conditional) else None,
                "mean_turnover": float(part.turnover.mean()) if len(part) else None,
                "total_commissions": float(part.commission.sum()) if len(part) else None,
                "mean_lambda": float(part.lambda_t.mean()) if len(part) else None,
                "cagr": None, "maximum_drawdown": None,
                "path_metrics_status": "not_defined_for_discontinuous_regime_subset"})
    per_run = pd.DataFrame(per_run)
    regime_lookup = {(r['mode'], r['seed'], r['regime']): r for r in regimes}
    mechanism = []
    for seed in seeds:
        delta = {}
        for regime in ('high', 'low'):
            a = regime_lookup[('adaptive', seed, regime)]['mean_turnover']
            b = regime_lookup[('fixed', seed, regime)]['mean_turnover']
            delta[regime] = a - b if a is not None and b is not None else None
        mechanism.append({'seed': seed, 'adaptive_minus_fixed_high': delta['high'],
            'adaptive_minus_fixed_low': delta['low'], 'high_minus_low_interaction':
            delta['high'] - delta['low'] if all(v is not None for v in delta.values()) else None})
    paired = []
    for left, right in PAIRS:
        for seed in seeds:
            a = per_run[(per_run['mode'] == left) & (per_run.seed == seed)].iloc[0]
            b = per_run[(per_run['mode'] == right) & (per_run.seed == seed)].iloc[0]
            for metric in METRICS:
                valid = pd.notna(a[metric]) and pd.notna(b[metric])
                paired.append({"comparison": left + "_minus_" + right, "seed": seed, "metric": metric,
                    "left": a[metric] if pd.notna(a[metric]) else None,
                    "right": b[metric] if pd.notna(b[metric]) else None,
                    "difference": float(a[metric] - b[metric]) if valid else None,
                    "higher_is_better": metric in {"net_sharpe", "cagr"}})
    paired = pd.DataFrame(paired)
    seed_summary = []
    for (comparison, metric), group in paired.groupby(["comparison", "metric"], sort=True):
        v = group.difference
        complete = bool(v.notna().all())
        seed_summary.append({"comparison": comparison, "metric": metric, "seeds": len(group),
            "mean_paired_difference": float(v.mean()) if complete else None,
            "sample_std_paired_difference": float(v.std(ddof=1)) if complete and len(group) > 1 else None,
            "positive_differences": int((v > 0).sum()), "negative_differences": int((v < 0).sum()),
            "status": "defined" if complete else "undefined_metric_in_seed"})
    returns = np.stack([records[k].net_return.to_numpy(float) for k in keys], axis=1)
    turnover = np.stack([records[k].turnover.to_numpy(float) for k in keys], axis=1)
    if not np.isfinite(returns).all() or not np.isfinite(turnover).all() or not np.isfinite(sigma).all():
        raise ValueError("Nonfinite analysis input")
    indices_by_mode = {m: [keys.index((m, s)) for s in seeds] for m in modes}

    def statistics(indices):
        r, t, sig = returns[indices], turnover[indices], sigma[indices]
        masks = {"overall": np.ones(len(indices), dtype=bool), "high": sig > sigma_ref, "low": sig <= sigma_ref}
        result = {}
        for regime, mask in masks.items():
            sh = _sharpe(r[mask])
            for left, right in PAIRS:
                result[(left + "_minus_" + right, regime, "net_sharpe")] = float(np.mean(
                    sh[indices_by_mode[left]] - sh[indices_by_mode[right]]))
        differences = {}
        for regime in ("high", "low"):
            mask = masks[regime]
            differences[regime] = (float(np.mean(t[mask][:, indices_by_mode['adaptive']] -
                t[mask][:, indices_by_mode['fixed']])) if mask.any() else float('nan'))
            result[("adaptive_minus_fixed", regime, "mean_turnover")] = differences[regime]
        result[("adaptive_minus_fixed", "high_minus_low", "turnover_interaction")] = differences['high'] - differences['low']
        return result

    point = statistics(np.arange(len(dates)))
    intervals = []
    if (bootstrap.get("method") != "paired_stationary_circular" or bootstrap.get("seed_resampling") is not False
            or bootstrap.get("interval") != "percentile" or type(bootstrap.get("resamples")) is not int
            or bootstrap['resamples'] < 2 or not 0 < bootstrap['confidence'] < 1):
        raise ValueError("Unsupported bootstrap protocol")
    for block in bootstrap['expected_block_lengths']:
        rng = np.random.default_rng(bootstrap['seed'])
        draws = {k: [] for k in point}
        for _ in range(bootstrap['resamples']):
            # One common draw preserves pairing across every mode and seed.
            for key, value in statistics(stationary_indices(len(dates), block, rng)).items():
                draws[key].append(value)
        for key, values in draws.items():
            array = np.asarray(values)
            valid = bool(np.isfinite(array).all() and np.isfinite(point[key]))
            tail = (1. - bootstrap['confidence']) / 2.
            bounds = np.quantile(array, [tail, 1 - tail]) if valid else [None, None]
            intervals.append({"comparison": key[0], "regime": key[1], "metric": key[2],
                "point": point[key] if np.isfinite(point[key]) else None,
                "block_length": block, "resamples": len(array), "invalid_resamples": int((~np.isfinite(array)).sum()),
                "lower": float(bounds[0]) if valid else None, "upper": float(bounds[1]) if valid else None,
                "status": "computed_conditional_time_interval" if valid else "undefined_no_replications_discarded",
                "uncertainty": "market_time_conditional_on_fixed_seeds_and_policies"})
    return {"metrics_by_seed": per_run, "paired_metrics": paired,
            "seed_stability": pd.DataFrame(seed_summary), "regime_metrics": pd.DataFrame(regimes),
            "turnover_mechanism_by_seed": pd.DataFrame(mechanism),
            "bootstrap_intervals": pd.DataFrame(intervals)}
