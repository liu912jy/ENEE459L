
from __future__ import annotations

from typing import Any, Iterable, Sequence

from schema import LATENCY_PERCENTILES, REL_FLOOR

import json
from pathlib import Path

########################################################
##########   Helper functions         ##################
########################################################


def deviation_sentence(abs_res: dict[str, Any], rel_res: dict[str, Any]) -> str:
    """The two deviation metrics in one line, with the caveat. Provided complete.

    Read it before writing `max_rel_dev`. The clause after the semicolon is the
    honest disclosure and it appears whenever any denominator was floored --
    which, on a classification head, is almost always.
    """
    line = (f"max |a-b| = {abs_res['max_abs']:.6g} at index {abs_res['index']}; "
            f"max relative = {rel_res['max_rel']:.6g}")
    if rel_res["floored"]:
        line += (f" (denominator floored on {rel_res['floored']} of "
                 f"{rel_res['n']} entries; smallest |b| was "
                 f"{rel_res['denom_min']:.3g})")
    return line


def decision_sentence(dm: dict[str, Any], tol: float) -> str:
    """The claim `decision_margin` licenses, or the one it refuses. Provided."""
    if dm["safe"]:
        return (f"no decision flips: the smallest margin over {dm['n']} inputs "
                f"is {dm['min']:.6g} and the deviation is {tol:.6g}")
    return (f"{dm['below_tol']} of {dm['n']} decisions sit inside the "
            f"deviation: smallest margin {dm['min']:.6g} at row {dm['argmin']} "
            f"against a deviation of {tol:.6g}")


class AgreementError(ValueError):
    """A comparison this module cannot make. `engine.py` exits 3."""


def _finite(v: Any, where: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise AgreementError(f"{where} is {v!r}, which is not a number.")
    f = float(v)
    if f != f or f in (float("inf"), float("-inf")):
        raise AgreementError(
            f"{where} is {f}. A NaN or an infinity in an output array is the "
            "run failing quietly; it is not a value to take a maximum over."
        )
    return f


def _percentile(xs: Sequence[float], q: float) -> float:
    """Linear interpolation between the two nearest ranks. Provided complete.

    The same implementation Labs 04 through 08 use, here for the same reason:
    a lab whose percentile is nearest-rank and a lab whose percentile
    interpolates produce different p95s from the same samples, and a student
    comparing this week's table against last week's should be comparing the
    board and not the arithmetic.
    """
    s = sorted(float(x) for x in xs)
    if not s:
        raise AgreementError("no samples.")
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * (float(q) / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] + (s[hi] - s[lo]) * frac


########################################################
##########    Start from here          #################
########################################################


def latency_report(ms: Iterable[float]) -> dict[str, Any]:
    """Get the latency information from the given times."""
    times = []

    for t in ms:
        t = _finite(t, "latency")
        if t < 0:
            raise AgreementError("Latency cannot be negative")
        times.append(t)

    if len(times) == 0:
        raise AgreementError("No data")

    n = len(times)
    avg = sum(times) / n
    worst = max(times)

    p50 = _percentile(times, 50)
    p95 = _percentile(times, 95)
    p99 = _percentile(times, 99)

    result = {
        "n": n,
        "mean": avg,
        "worst": worst,
        "p50": p50,
        "p95": p95,
        "p99": p99,
        "jitter": p95 - p50
    }

    return result


def speedup(base_ms: float, other_ms: float) -> dict[str, Any]:
    """Calculate how much faster the second time is."""
    base = _finite(base_ms, "base")
    other = _finite(other_ms, "other")

    if base <= 0 or other <= 0:
        raise AgreementError("Time must be greater than zero")

    factor = base / other
    saved = base - other

    if factor > 1:
        faster = True
    else:
        faster = False

    return {
        "factor": factor,
        "faster": faster,
        "saved_ms": saved
    }


def max_abs_dev(a: Sequence[float], b: Sequence[float]) -> dict[str, Any]:
    """Find the biggest difference between two lists."""
    if len(a) == 0 or len(a) != len(b):
        raise AgreementError("Lists have wrong lengths")

    list_a = []
    list_b = []

    for x in a:
        list_a.append(_finite(x, "a"))

    for x in b:
        list_b.append(_finite(x, "b"))

    biggest = 0
    index = 0

    for i in range(len(list_a)):
        difference = abs(list_a[i] - list_b[i])

        if difference > biggest:
            biggest = difference
            index = i

    return {
        "max_abs": biggest,
        "index": index,
        "a": list_a[index],
        "b": list_b[index],
        "n": len(list_a)
    }


def max_rel_dev(a: Sequence[float], b: Sequence[float],
                floor: float = REL_FLOOR) -> dict[str, Any]:
    """Find the biggest relative difference between two lists."""
    floor = _finite(floor, "floor")

    if floor <= 0:
        raise AgreementError("Invalid floor")

    if len(a) == 0 or len(a) != len(b):
        raise AgreementError("Lists have wrong lengths")

    list_a = []
    list_b = []

    for x in a:
        list_a.append(_finite(x, "a"))

    for x in b:
        list_b.append(_finite(x, "b"))

    biggest = -1
    index = 0
    floored = 0
    smallest = abs(list_b[0])

    for i in range(len(list_a)):
        denominator = abs(list_b[i])

        if denominator < smallest:
            smallest = denominator

        if denominator < floor:
            floored += 1
            denominator = floor

        difference = abs(list_a[i] - list_b[i])
        relative = difference / denominator

        if relative > biggest:
            biggest = relative
            index = i

    return {
        "max_rel": biggest,
        "index": index,
        "floored": floored,
        "denom_min": smallest,
        "n": len(list_a)
    }


def top1_agreement(rows_a: Sequence[Sequence[float]],
                   rows_b: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Check if two models predict the same classes."""
    if len(rows_a) == 0 or len(rows_a) != len(rows_b):
        raise AgreementError("Wrong number of rows")

    agree = 0
    first = None

    for i in range(len(rows_a)):
        a = []
        b = []

        for x in rows_a[i]:
            a.append(_finite(x, "a"))

        for x in rows_b[i]:
            b.append(_finite(x, "b"))

        if len(a) == 0 or len(a) != len(b):
            raise AgreementError("Wrong row size")

        best_a = a.index(max(a))
        best_b = b.index(max(b))

        if best_a == best_b:
            agree += 1
        else:
            if first is None:
                first = i

    n = len(rows_a)

    return {
        "n": n,
        "agree": agree,
        "disagree": n - agree,
        "rate": agree / n,
        "first_disagreement": first
    }


def decision_margin(rows: Sequence[Sequence[float]],
                    tol: float) -> dict[str, Any]:
    """Check the difference between the two highest scores."""
    tol = _finite(tol, "tol")

    if tol < 0:
        raise AgreementError("Invalid tolerance")

    if len(rows) == 0:
        raise AgreementError("No rows")

    margins = []

    for row in rows:
        scores = []

        for x in row:
            scores.append(_finite(x, "score"))

        if len(scores) < 2:
            raise AgreementError("Not enough scores")

        scores.sort(reverse=True)
        margin = scores[0] - scores[1]
        margins.append(margin)

    smallest = min(margins)
    index = margins.index(smallest)
    avg = sum(margins) / len(margins)
    median = _percentile(margins, 50)

    below = 0

    for m in margins:
        if m < tol:
            below += 1

    if below == 0:
        safe = True
    else:
        safe = False

    return {
        "n": len(margins),
        "min": smallest,
        "median": median,
        "mean": avg,
        "argmin": index,
        "below_tol": below,
        "safe": safe
    }


#############################################################
###############  Main function starts here ##################
#############################################################

def main(
    input_path: str | Path = "benchmark_input_data.json",
    output_path: str | Path = "benchmark_results.json",
) -> None:
    in_file = Path(input_path)
    out_file = Path(output_path)

    # 1. Load synthetic data from JSON
    if not in_file.exists():
        raise FileNotFoundError(f"Input file not found: {in_file.resolve()}")

    with in_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    timings_base = data["timings_base"]
    timings_optimized = data["timings_optimized"]
    flat_logits_base = data["flat_logits_base"]
    flat_logits_opt = data["flat_logits_opt"]
    batched_scores_base = data["batched_scores_base"]
    batched_scores_opt = data["batched_scores_opt"]

    # 2. Execute metric and report functions
    report_base = latency_report(timings_base)
    report_opt = latency_report(timings_optimized)
    perf_speedup = speedup(base_ms=report_base["p50"], other_ms=report_opt["p50"])

    abs_dev = max_abs_dev(flat_logits_opt, flat_logits_base)
    rel_dev = max_rel_dev(flat_logits_opt, flat_logits_base, floor=REL_FLOOR)

    agreement = top1_agreement(batched_scores_opt, batched_scores_base)
    margin = decision_margin(batched_scores_base, tol=abs_dev["max_abs"])

    # 3. Structure output results
    results: dict[str, Any] = {
        "latency": {
            "base": report_base,
            "optimized": report_opt,
            "speedup_p50": perf_speedup,
        },
        "deviations": {
            "max_abs_dev": abs_dev,
            "max_rel_dev": rel_dev,
            "summary_sentence": deviation_sentence(abs_dev, rel_dev),
        },
        "agreement": {
            "top1_agreement": agreement,
            "decision_margin": margin,
            "decision_sentence": decision_sentence(margin, tol=abs_dev["max_abs"]),
        },
    }

    # 4. Save results to JSON
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # 5. Print a console summary
    print(f"Loaded input from: {in_file.resolve()}")
    print(f"Results saved to:  {out_file.resolve()}\n")
    print("--- Summary ---")
    print(results["deviations"]["summary_sentence"])
    print(results["agreement"]["decision_sentence"])
    print(
        f"Speedup (p50): {perf_speedup['factor']:.2f}x "
        f"({perf_speedup['saved_ms']:.2f} ms saved)"
    )

if __name__ == "__main__":
    main()