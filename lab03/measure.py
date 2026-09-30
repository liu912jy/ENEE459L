from __future__ import annotations

import statistics
from typing import Any

from bench import Bench, measured, read_first, read_text, unknown

import json

# A sample is still warm-up while it exceeds the settled rate by this fraction.
WARMUP_TOL = 0.5

# How many samples must sit strictly above a quantile before that quantile is an
# estimate rather than "the biggest number we saw, wearing a hat".
MIN_SAMPLES_ABOVE = 5

# Percentiles the record carries, in the order the schema lists them.
PERCENTILES = (50, 95, 99)

# The widest gap between neighbouring measurements, as a multiple of the typical
# gap, beyond which the sample is treated as coming from two populations.
MULTIMODAL_GAP_RATIO = 20.0

# Neither side of that gap is a mode unless it holds at least this fraction.
MIN_MODE_FRACTION = 0.10

# Below this many retained samples, modality is not a question worth answering.
MIN_SAMPLES_FOR_MODALITY = 20

# How far the last third of a run may drift from the first third, relative to
# the run's own median, before the run is not one population either.
STATIONARITY_TOL = 0.10
MIN_SAMPLES_FOR_STATIONARITY = 12

THERMAL_ZONES = "sys/devices/virtual/thermal"

POWER_RAIL_CANDIDATES = (
    "sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon3/in1_input",
    "sys/bus/i2c/drivers/ina3221/1-0040/iio:device0/in_power0_input",
    "sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power0_input",
)

GPU_LOAD_CANDIDATES = (
    "sys/devices/platform/gpu.0/load",
    "sys/devices/gpu.0/load",
)

CPUFREQ_MIN = "sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq"
CPUFREQ_MAX = "sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq"


# ===========================================================================
# 1. The loop
# ===========================================================================
def run_timed_iterations(bench: Bench, repeats: int = 100) -> list[float]:
    samples = []
    bench.workload.synchronize()
    
    for _ in range(repeats):
        start = bench.clock()
        bench.workload.run()
        bench.workload.synchronize()
        end = bench.clock()
        samples.append((end - start)/1_000_000.0)
        
    return samples


def find_warmup_boundary(samples: list[float]) -> dict[str, Any]:
    source = ("leading prefix above (1+0.5) x median of the run's second half")
    
    if len(samples) < 4:
        return unknown(source, "there are too few samples")
    
    settled = statistics.median(samples[len(samples)//2:])
    
    if settled <= 0:
        return unknown(source, "the median of the second half is non-positive")
    
    threshold = settled * (1+WARMUP_TOL)
    
    discarded = 0
    for sample in samples:
        if sample > threshold:
            discarded += 1
        else:
            break
  
    return measured(
        discarded,
        source,
        settled_rate_ms=round(settled, 4),
        threshold_ms=round(threshold, 4),
        tolerance=WARMUP_TOL,
        retained=len(samples) - discarded,
    )


def summarize(samples: list[float]) -> dict[str, Any]:
    keys = ("mean", "std", "min", "max", "p50", "p95", "p99")

    if not samples:
        return {"n": 0, **{key: None for key in keys}}

    s = sorted(samples)
    n = len(s)

    percentiles = {}
    for q in (50, 95, 99):
        h = (n - 1) * q / 100
        i = int(h)

        if i >= n - 1:
            value =s[-1]
        else:
            value = s[i]+(h-i)*(s[i+1]- s[i])

        percentiles[q] = round(value, 4)

    return {
        "n": n,
        "mean": round(statistics.fmean(s), 4),
        "std": round(statistics.stdev(s), 4) if n > 2 else 0.0,
        "min": round(s[0], 4),
        "max": round(s[-1], 4),
        "p50": percentiles[50],
        "p95": percentiles[95],
        "p99": percentiles[99],
    }

def is_multimodal(samples: list[float]) -> dict[str, Any]:
    
    source = ("widest trimmed gap >= 20.0x the median gap, "
        "with >= 10% of samples on each side")

    n = len(samples)

    if n < MIN_SAMPLES_FOR_MODALITY:
        return unknown(source, "there are not enough samples")

    s = sorted(samples)
    trim = int(n * 0.05)
    trimmed = s[trim:n - trim]

    gaps = [
        trimmed[i + 1] - trimmed[i]
        for i in range(len(trimmed) - 1)
    ]

    typical_gap = statistics.median(gaps)

    if typical_gap <= 0:
        return unknown(source, "the timer resolution is too coarse")

    gap_index = max(range(len(gaps)), key=lambda i: gaps[i])
    widest_gap = gaps[gap_index]
    ratio = widest_gap / typical_gap

    split = trim + gap_index + 1
    left = s[:split]
    right = s[split:]

    multimodal = (
        ratio >= MULTIMODAL_GAP_RATIO
        and len(left) / n >= MIN_MODE_FRACTION
        and len(right) / n >= MIN_MODE_FRACTION
    )

    return measured(
        multimodal,
        source,
        gap_ratio=round(ratio, 2),
        widest_gap_ms=round(widest_gap, 4),
        typical_gap_ms=round(typical_gap, 5),
        modes=[
            {
                "n": len(left),
                "share": round(len(left) / n, 4),
                "median_ms": round(statistics.median(left), 4),
            },
            {
                "n": len(right),
                "share": round(len(right) / n, 4),
                "median_ms": round(statistics.median(right), 4),
            },
        ],
    )

# ===========================================================================
# 7. The clock ceiling the run happened under
# ===========================================================================


def probe_power_state(bench: Bench) -> dict[str, Any]:
    source = "nvpmodel -q"
    result = bench.runner(["nvpmodel", "-q"])

    if not result.ok or result.returncode != 0:
        detail = (
            result.error
            or result.stdout.strip()
            or f"command exited with code {result.returncode}"
        )
        return unknown(source, detail)

    lines = result.stdout.splitlines()
    mode_name = None
    mode_index = None

    for i, line in enumerate(lines):
        if "NV Power Mode:" in line:
            mode_name = line.split("NV Power Mode:", 1)[1].strip()

            if i + 1 < len(lines):
                try:
                    mode_index = int(lines[i + 1].strip())
                except ValueError:
                    pass
                
            break

    if not mode_name or mode_index is None:
        return unknown(source, "could not parse power mode name and index")

    findings = measured(mode_name, source, mode_index=mode_index)

    minimum = read_text(bench.telemetry, CPUFREQ_MIN)
    maximum = read_text(bench.telemetry, CPUFREQ_MAX)
    clocks_source = f"{CPUFREQ_MIN} vs {CPUFREQ_MAX}"

    try:
        minimum_freq = int(minimum) if minimum is not None else None
        maximum_freq = int(maximum) if maximum is not None else None
        
    except ValueError:
        minimum_freq = maximum_freq = None

    if minimum_freq is None or maximum_freq is None:
        findings["jetson_clocks"] = None
        findings["jetson_clocks_source"] = unknown(
            clocks_source,
            "CPU frequency limits could not be read or parsed",
        )
        
    else:
        findings["jetson_clocks"] = minimum_freq == maximum_freq
        findings["jetson_clocks_source"] = measured(
            f"scaling_min_freq={minimum_freq}, "
            f"scaling_max_freq={maximum_freq}",
            clocks_source,
        )

    return findings



def probe_telemetry(bench: Bench) -> dict[str, Any]:
    temperature_source = f"{THERMAL_ZONES}/*/temp"
    temperatures = []

    thermal_root = bench.telemetry / THERMAL_ZONES

    try:
        zones = sorted(thermal_root.glob("thermal_zone*"))
    except OSError:
        zones = []

    for zone in zones:
        relative_zone = f"{THERMAL_ZONES}/{zone.name}"
        
        try:
            raw = read_text(bench.telemetry, f"{relative_zone}/temp")
        except (OSError, TypeError):
            continue

        if raw is None:
            continue

        try:
            raw_temperature = int(raw)
        except ValueError:
            continue

        if raw_temperature <= -1000:
            continue

        zone_name = (
            read_text(bench.telemetry, f"{relative_zone}/type")
            or zone.name
        )

        temperatures.append((raw_temperature / 1000.0, zone_name))

    if temperatures:
        hottest, zone_name = max(temperatures, key=lambda item: item[0])
        temperature = measured(
            round(hottest, 4),
            temperature_source,
            zone=zone_name,
            zones_read=len(temperatures),
        )
    else:
        temperature = unknown(
            temperature_source,
            "no valid thermal zones could be read",
        )

    power_reading = read_first(bench.telemetry, POWER_RAIL_CANDIDATES)

    if power_reading is None:
        power = unknown(
            " | ".join(POWER_RAIL_CANDIDATES),
            "none of the documented INA3221 rail paths could be read",
        )
    else:
        path, raw = power_reading
        try:
            power = measured(int(raw), path)
        except ValueError:
            power = unknown(path, "power reading is not an integer")

    gpu_reading = read_first(bench.telemetry, GPU_LOAD_CANDIDATES)

    if gpu_reading is None:
        gpu = unknown(
            " | ".join(GPU_LOAD_CANDIDATES),
            "none of the documented GPU load paths could be read",
        )
    else:
        path, raw = gpu_reading
        try:
            gpu = measured(
                round(int(raw) / 10.0, 4),
                path,
                units="per-mille / 10",
            )
        except ValueError:
            gpu = unknown(path, "GPU load reading is not an integer")

    return {
        "temperature_c": temperature,
        "power_mw": power,
        "gpu_utilization_percent": gpu,
    }
    

## for debugging - uncomment the following lines for debugging.
# if __name__ == "__main__":
    # env = Bench.real()
    # out = find_warmup_boundary(samples)
    # print(out)

# for generating system_report.json
if __name__ == "__main__":
    # calling base environment
    env = Bench.real()

    # get your samples
    samples = run_timed_iterations(env, repeats=100)

    # testing measurments and probes
    report = {
        "warmup_boundary": find_warmup_boundary(samples),
        "summarize_setup": summarize(samples),
        "is_multimodal": is_multimodal(samples),
        "probe_power_state": probe_power_state(env),
        "probe_telemetry": probe_telemetry(env),
    }

    # save samples
    path = "samples_analysis.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=4)

    # save report
    path = "system_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)