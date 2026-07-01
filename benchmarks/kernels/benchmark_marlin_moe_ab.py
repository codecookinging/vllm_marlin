#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A/B microbenchmark for Marlin MoE kernel changes.

Benchmarks ``vllm.model_executor.layers.fused_moe.experts.marlin_moe.
fused_marlin_moe`` with fixed synthetic workloads and reports per-case
latency stats using CUDA events.

Timing modes (``--timing``):
- event (default): one CUDA event pair per repeat; reports mean/p50/p95/std.
- batch: one event pair around all repeats; reports average latency (fast
  sanity checks / stable prefill regression).

Default workloads target zai-org/GLM-5 at tensor-parallel size 8
(K=6144, N=256, E=256, topk=8). Use ``--model generic`` for legacy shapes.

Only ``act_order=False`` and ``is_k_full=True`` cases are supported locally.

Modes:
1) single: benchmark one checkout
2) ab: compare baseline vs candidate roots via subprocess

Example::

    .venv/bin/python benchmarks/kernels/benchmark_marlin_moe_ab.py --list-cases
    .venv/bin/python benchmarks/kernels/benchmark_marlin_moe_ab.py \\
        --mode single --model glm5 --repeat 200 --cases glm5_decode_m1
    .venv/bin/python benchmarks/kernels/benchmark_marlin_moe_ab.py \\
        --mode single --model glm5 --timing batch --repeat 10000 \\
        --cases glm5_prefill_m512
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

BENCHMARK_SUITE_ID = "glm5-tp8-v3"
LEGACY_CASE_NAMES = {
    "small_m_act_off_colwise",
    "med_m_act_off_group128",
    "med_m_act_on_group128",
    "med_m_large_k_act_on",
    "large_m_act_off_group128",
    "large_n_act_on_group128",
}

_REPO_ROOT = Path(__file__).resolve().parents[2]

torch = None


@dataclasses.dataclass(frozen=True)
class MarlinMoECase:
    name: str
    m: int
    n: int
    k: int
    e: int
    topk: int
    group_size: int
    act_order: bool
    is_k_full: bool = True
    dtype: str = "float16"
    weight: float | None = None


@dataclasses.dataclass(frozen=True)
class ThresholdConfig:
    min_case_p50_speedup: float = 1.0
    min_weighted_p50_speedup: float = 1.0
    max_case_p95_ratio: float = 1.05
    max_weighted_p95_ratio: float = 1.05


@dataclasses.dataclass(frozen=True)
class ModelPreset:
    name: str
    k: int
    n: int
    e: int
    topk: int
    tp_size: int = 1
    group_size: int = 128
    act_order: bool = False
    dtype: str = "bfloat16"
    decode_ms: tuple[int, ...] = (1, 2, 4, 8)
    small_batch_ms: tuple[int, ...] = (16, 32, 64)
    prefill_ms: tuple[int, ...] = (128, 256, 512, 1024, 2048, 4096, 8192)


_GLM5_K = 6144
_GLM5_N = 2048
_GLM5_E = 256
_GLM5_TOPK = 8
_GLM5_TP = 8

MODEL_PRESETS: dict[str, ModelPreset] = {
    "glm5": ModelPreset(
        name="glm5",
        k=_GLM5_K,
        n=_GLM5_N // _GLM5_TP,
        e=_GLM5_E,
        topk=_GLM5_TOPK,
        tp_size=_GLM5_TP,
    ),
    "glm5_tp1": ModelPreset(
        name="glm5_tp1",
        k=_GLM5_K,
        n=_GLM5_N,
        e=_GLM5_E,
        topk=_GLM5_TOPK,
        tp_size=1,
    ),
    "glm5_ep8": ModelPreset(
        name="glm5_ep8",
        k=_GLM5_K,
        n=_GLM5_N // _GLM5_TP,
        e=_GLM5_E // 8,
        topk=_GLM5_TOPK,
        tp_size=_GLM5_TP,
    ),
}

GENERIC_CASES: list[MarlinMoECase] = [
    MarlinMoECase(
        name="small_m_act_off_colwise",
        m=64,
        n=4096,
        k=4096,
        e=64,
        topk=2,
        group_size=-1,
        act_order=False,
    ),
    MarlinMoECase(
        name="med_m_act_off_group128",
        m=256,
        n=8192,
        k=4096,
        e=64,
        topk=2,
        group_size=128,
        act_order=False,
    ),
    MarlinMoECase(
        name="med_m_act_on_group128",
        m=256,
        n=8192,
        k=4096,
        e=64,
        topk=2,
        group_size=128,
        act_order=True,
    ),
    MarlinMoECase(
        name="med_m_large_k_act_on",
        m=192,
        n=8192,
        k=8192,
        e=64,
        topk=2,
        group_size=128,
        act_order=True,
    ),
    MarlinMoECase(
        name="large_m_act_off_group128",
        m=512,
        n=8192,
        k=8192,
        e=64,
        topk=2,
        group_size=128,
        act_order=False,
    ),
    MarlinMoECase(
        name="large_n_act_on_group128",
        m=256,
        n=14336,
        k=4096,
        e=64,
        topk=2,
        group_size=128,
        act_order=True,
    ),
]


def _lazy_imports() -> None:
    global torch
    if torch is not None:
        return
    import torch as _torch

    torch = _torch


def _set_random_seed(seed: int) -> None:
    _lazy_imports()
    from vllm.utils.torch_utils import set_random_seed

    set_random_seed(seed)


def _make_glm5_case(
    preset: ModelPreset,
    *,
    name: str,
    m: int,
    k: int | None = None,
    act_order: bool | None = None,
    is_k_full: bool = True,
    weight: float | None = None,
) -> MarlinMoECase:
    case_k = preset.k if k is None else k
    return MarlinMoECase(
        name=name,
        m=m,
        n=preset.n,
        k=case_k,
        e=preset.e,
        topk=preset.topk,
        group_size=preset.group_size,
        act_order=preset.act_order if act_order is None else act_order,
        is_k_full=is_k_full,
        dtype=preset.dtype,
        weight=weight,
    )


def _glm5_core_cases(preset: ModelPreset) -> list[MarlinMoECase]:
    cases: list[MarlinMoECase] = []
    for m in preset.decode_ms:
        cases.append(
            _make_glm5_case(
                preset,
                name=f"{preset.name}_decode_m{m}",
                m=m,
                weight=float(m * preset.n * preset.k * preset.topk),
            )
        )
    for m in preset.small_batch_ms:
        cases.append(_make_glm5_case(preset, name=f"{preset.name}_small_m{m}", m=m))
    for m in preset.prefill_ms:
        cases.append(_make_glm5_case(preset, name=f"{preset.name}_prefill_m{m}", m=m))
    return cases


def _glm5_full_cases(preset: ModelPreset) -> list[MarlinMoECase]:
    cases = _glm5_core_cases(preset)
    for m in (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192):
        if any(c.m == m and c.k == preset.k for c in cases):
            continue
        cases.append(_make_glm5_case(preset, name=f"{preset.name}_sweep_m{m}", m=m))
    return cases


def cases_for_model(model: str) -> list[MarlinMoECase]:
    if model == "generic":
        return list(GENERIC_CASES)
    if model == "glm5_full":
        return _glm5_full_cases(MODEL_PRESETS["glm5"])
    if model not in MODEL_PRESETS:
        raise ValueError(
            f"Unknown model preset: {model}. "
            f"Available: {sorted(MODEL_PRESETS)} + generic, glm5_full"
        )
    return _glm5_core_cases(MODEL_PRESETS[model])


DEFAULT_CASES: list[MarlinMoECase] = cases_for_model("glm5")


def _describe_preset(model: str) -> str:
    if model == "generic":
        return "legacy generic Mixtral-like shapes"
    if model not in MODEL_PRESETS:
        return model
    preset = MODEL_PRESETS[model]
    return (
        f"K={preset.k}, N={preset.n}, E={preset.e}, topk={preset.topk}, "
        f"tp={preset.tp_size}, group={preset.group_size}, "
        f"act_order={preset.act_order}, dtype={preset.dtype}"
    )


def _print_case_plan(model: str, cases: list[MarlinMoECase]) -> None:
    print(f"Benchmark suite: {BENCHMARK_SUITE_ID}")
    print(f"Model preset: {model} ({_describe_preset(model)})")
    print(f"Case count: {len(cases)}")
    print("-" * 132)
    print(
        f"{'case':36} {'M':>6} {'N':>6} {'K':>6} {'E':>4} "
        f"{'topk':>4} {'group':>6} {'act':>4} {'kfull':>5} {'dtype':>10}"
    )
    for case in cases:
        print(
            f"{case.name:36} {case.m:6d} {case.n:6d} {case.k:6d} {case.e:4d} "
            f"{case.topk:4d} {case.group_size:6d} "
            f"{int(case.act_order):4d} {int(case.is_k_full):5d} {case.dtype:>10}"
        )
    print("-" * 132)


def _validate_case_selection(model: str, cases: list[MarlinMoECase]) -> None:
    unsupported = [
        case.name for case in cases if case.act_order or not case.is_k_full
    ]
    if unsupported:
        raise ValueError(
            "Local benchmark supports act_order=False and is_k_full=True only. "
            f"Unsupported cases: {unsupported}"
        )
    if model == "generic":
        return
    legacy_hits = [case.name for case in cases if case.name in LEGACY_CASE_NAMES]
    if legacy_hits:
        raise ValueError(
            "Legacy generic cases were selected while --model "
            f"{model!r} is active: {legacy_hits}. "
            "Drop --cases or use --model generic."
        )
    if model.startswith("glm5") and cases:
        sample = cases[0]
        if sample.topk != _GLM5_TOPK:
            raise ValueError(
                f"Selected cases do not match GLM-5 shape expectations: "
                f"got topk={sample.topk}."
            )
        expected_n = MODEL_PRESETS.get(model, MODEL_PRESETS["glm5"]).n
        if sample.n != expected_n:
            raise ValueError(
                f"Selected cases do not match GLM-5 N expectation for {model}: "
                f"got N={sample.n}."
            )


def _resolve_benchmark_script(root: Path) -> Path:
    candidates = [
        root / "benchmarks/kernels/benchmark_marlin_moe_ab.py",
        root / "benchmark_marlin_moe_ab.py",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return Path(__file__).resolve()


def _require_moe_extension() -> None:
    _lazy_imports()
    import vllm._custom_ops as ops
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
        fused_marlin_moe,
    )

    if not hasattr(ops, "moe_wna16_marlin_gemm"):
        raise RuntimeError(
            "moe_wna16_marlin_gemm is not available. "
            "Build vLLM with CUDA Marlin MoE support "
            "(uv pip install -e . --torch-backend=auto)."
        )
    print(
        "MoE kernel: "
        f"{fused_marlin_moe.__module__}.{fused_marlin_moe.__qualname__}"
    )


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return math.nan
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    alpha = pos - lo
    return sorted_values[lo] * (1.0 - alpha) + sorted_values[hi] * alpha


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def _case_weight(case: dict[str, Any]) -> float:
    if case.get("weight") is not None:
        return float(case["weight"])
    return float(case["m"] * case["n"] * case["k"] * case["topk"])


def _quantize_expert_stack(
    w: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
        marlin_quantize,
    )
    from vllm.scalar_type import scalar_types

    quant_type = scalar_types.uint4b8
    k = w.shape[-1]
    qweight_l: list[torch.Tensor] = []
    scales_l: list[torch.Tensor] = []
    for expert_id in range(w.shape[0]):
        perm = torch.arange(k, device=w.device)
        _w_ref, qweight, scales, _g_idx, _sort_indices, _ = marlin_quantize(
            w[expert_id].transpose(1, 0),
            quant_type,
            group_size,
            False,
            perm,
        )
        qweight_l.append(qweight)
        scales_l.append(scales)
    return torch.stack(qweight_l, dim=0).contiguous(), torch.stack(scales_l, dim=0)


def _build_case_tensors(case: MarlinMoECase, seed: int) -> dict[str, Any]:
    _lazy_imports()
    from vllm.model_executor.layers.fused_moe import fused_topk
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
        fused_marlin_moe,
    )
    from vllm.scalar_type import scalar_types

    _set_random_seed(seed)
    dtype = _dtype_from_name(case.dtype)
    device = torch.device("cuda")

    hidden_states = torch.randn((case.m, case.k), device=device, dtype=dtype) / 10
    w1 = torch.randn((case.e, 2 * case.n, case.k), device=device, dtype=dtype) / 10
    w2 = torch.randn((case.e, case.k, case.n), device=device, dtype=dtype) / 10
    scores = torch.randn((case.m, case.e), device=device, dtype=dtype)

    w1_qweight, w1_scales = _quantize_expert_stack(w1, case.group_size)
    w2_qweight, w2_scales = _quantize_expert_stack(w2, case.group_size)
    topk_weights, topk_ids, _ = fused_topk(
        hidden_states, scores, case.topk, renormalize=False
    )

    return {
        "hidden_states": hidden_states,
        "w1_qweight": w1_qweight,
        "w2_qweight": w2_qweight,
        "w1_scales": w1_scales,
        "w2_scales": w2_scales,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "quant_type_id": scalar_types.uint4b8.id,
        "fused_marlin_moe": fused_marlin_moe,
        "is_k_full": case.is_k_full,
        "global_num_experts": case.e,
    }


def _run_one_case(
    case: MarlinMoECase,
    warmup: int,
    repeat: int,
    seed: int,
    timing: str,
) -> dict[str, Any]:
    _lazy_imports()
    with torch.inference_mode():
        return _run_one_case_impl(case, warmup, repeat, seed, timing)


def _run_one_case_impl(
    case: MarlinMoECase,
    warmup: int,
    repeat: int,
    seed: int,
    timing: str,
) -> dict[str, Any]:
    tensors = _build_case_tensors(case, seed=seed)

    def _kernel_call() -> torch.Tensor:
        return tensors["fused_marlin_moe"](
            hidden_states=tensors["hidden_states"],
            w1=tensors["w1_qweight"],
            w2=tensors["w2_qweight"],
            bias1=None,
            bias2=None,
            w1_scale=tensors["w1_scales"],
            w2_scale=tensors["w2_scales"],
            topk_weights=tensors["topk_weights"],
            topk_ids=tensors["topk_ids"],
            quant_type_id=tensors["quant_type_id"],
            global_num_experts=tensors["global_num_experts"],
            is_k_full=tensors["is_k_full"],
        )

    for _ in range(warmup):
        _kernel_call()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    if timing == "batch":
        start_event.record()
        for _ in range(repeat):
            _kernel_call()
        end_event.record()
        end_event.synchronize()
        total_us = start_event.elapsed_time(end_event) * 1000.0
        avg_us = total_us / repeat
        return {
            "case": dataclasses.asdict(case),
            "timing": timing,
            "samples_us": [],
            "total_us": total_us,
            "mean_us": avg_us,
            # A/B gates key off p50/p95; in batch mode they equal the average.
            "p50_us": avg_us,
            "p95_us": avg_us,
            "std_us": 0.0,
            "min_us": avg_us,
            "max_us": avg_us,
        }

    samples_us: list[float] = []
    for _ in range(repeat):
        start_event.record()
        _kernel_call()
        end_event.record()
        end_event.synchronize()
        samples_us.append(start_event.elapsed_time(end_event) * 1000.0)

    samples_sorted = sorted(samples_us)
    p50 = _percentile(samples_sorted, 0.50)
    p95 = _percentile(samples_sorted, 0.95)
    mean = statistics.fmean(samples_us)
    std = statistics.pstdev(samples_us) if len(samples_us) > 1 else 0.0
    return {
        "case": dataclasses.asdict(case),
        "timing": timing,
        "samples_us": samples_us,
        "mean_us": mean,
        "p50_us": p50,
        "p95_us": p95,
        "std_us": std,
        "min_us": samples_sorted[0],
        "max_us": samples_sorted[-1],
    }


def _run_single(
    cases: list[MarlinMoECase],
    warmup: int,
    repeat: int,
    seed: int,
    timing: str,
) -> dict[str, Any]:
    _lazy_imports()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark script.")
    _require_moe_extension()
    _ = torch.cuda.get_device_name(0)
    results = []
    for idx, case in enumerate(cases):
        result = _run_one_case(
            case, warmup=warmup, repeat=repeat, seed=seed + idx, timing=timing
        )
        results.append(result)
    return {
        "gpu_name": torch.cuda.get_device_name(0),
        "cuda_device_capability": torch.cuda.get_device_capability(0),
        "warmup": warmup,
        "repeat": repeat,
        "timing": timing,
        "results": results,
    }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _case_key(case_obj: dict[str, Any]) -> str:
    c = case_obj["case"]
    return (
        f"{c['name']}|m={c['m']}|n={c['n']}|k={c['k']}|e={c['e']}|"
        f"topk={c['topk']}|g={c['group_size']}|act={int(c['act_order'])}|"
        f"kfull={int(c['is_k_full'])}|dtype={c['dtype']}"
    )


def _print_single_report(report: dict[str, Any]) -> None:
    model = report.get("model", "?")
    timing = report.get("timing", "event")
    print(f"Benchmark suite: {report.get('benchmark_suite_id', BENCHMARK_SUITE_ID)}")
    print(f"Model preset: {model} ({report.get('model_description', '')})")
    print(
        f"GPU: {report['gpu_name']}  "
        f"SM: {tuple(report['cuda_device_capability'])}  "
        f"warmup={report['warmup']} repeat={report['repeat']} timing={timing}"
    )
    print("-" * 150)
    if timing == "batch":
        print(
            f"{'case':32} {'M':>5} {'N':>5} {'K':>5} {'E':>4} {'tk':>3} "
            f"{'avg(us)':>10} {'total(ms)':>10}"
        )
        for row in report["results"]:
            c = row["case"]
            total_ms = row.get("total_us", row["mean_us"] * report["repeat"]) / 1000.0
            print(
                f"{c['name']:32} {c['m']:5d} {c['n']:5d} {c['k']:5d} "
                f"{c['e']:4d} {c['topk']:3d} "
                f"{row['mean_us']:10.2f} {total_ms:10.2f}"
            )
        return

    print(
        f"{'case':32} {'M':>5} {'N':>5} {'K':>5} {'E':>4} {'tk':>3} "
        f"{'mean(us)':>10} {'p50(us)':>10} {'p95(us)':>10} {'std(us)':>10}"
    )
    for row in report["results"]:
        c = row["case"]
        print(
            f"{c['name']:32} {c['m']:5d} {c['n']:5d} {c['k']:5d} "
            f"{c['e']:4d} {c['topk']:3d} "
            f"{row['mean_us']:10.2f} {row['p50_us']:10.2f} "
            f"{row['p95_us']:10.2f} {row['std_us']:10.2f}"
        )


def _judge_case(
    base_row: dict[str, Any],
    cand_row: dict[str, Any],
    thresholds: ThresholdConfig,
) -> dict[str, Any]:
    case = base_row["case"]
    p50_speedup = base_row["p50_us"] / cand_row["p50_us"]
    p95_ratio = cand_row["p95_us"] / base_row["p95_us"]
    p50_ok = p50_speedup >= thresholds.min_case_p50_speedup
    p95_ok = p95_ratio <= thresholds.max_case_p95_ratio
    verdict = "PASS" if p50_ok and p95_ok else "FAIL"
    reasons: list[str] = []
    if not p50_ok:
        reasons.append(
            f"p50 speedup {p50_speedup:.3f}x < "
            f"{thresholds.min_case_p50_speedup:.3f}x"
        )
    if not p95_ok:
        reasons.append(
            f"p95 ratio {p95_ratio:.3f}x > {thresholds.max_case_p95_ratio:.3f}x"
        )
    return {
        "case_name": case["name"],
        "case_key": _case_key(base_row),
        "case": case,
        "weight": _case_weight(case),
        "base_p50_us": base_row["p50_us"],
        "cand_p50_us": cand_row["p50_us"],
        "base_p95_us": base_row["p95_us"],
        "cand_p95_us": cand_row["p95_us"],
        "p50_speedup": p50_speedup,
        "p95_ratio": p95_ratio,
        "p50_ok": p50_ok,
        "p95_ok": p95_ok,
        "verdict": verdict,
        "reason": "; ".join(reasons),
    }


def _build_ab_report(
    base_report: dict[str, Any],
    cand_report: dict[str, Any],
    thresholds: ThresholdConfig,
) -> dict[str, Any]:
    base_rows = {_case_key(r): r for r in base_report["results"]}
    cand_rows = {_case_key(r): r for r in cand_report["results"]}
    keys = sorted(set(base_rows.keys()) & set(cand_rows.keys()))
    if not keys:
        raise RuntimeError("No overlapping cases between baseline and candidate.")

    case_rows = [
        _judge_case(base_rows[key], cand_rows[key], thresholds) for key in keys
    ]

    weighted_base_p50 = sum(r["base_p50_us"] * r["weight"] for r in case_rows)
    weighted_cand_p50 = sum(r["cand_p50_us"] * r["weight"] for r in case_rows)
    weighted_base_p95 = sum(r["base_p95_us"] * r["weight"] for r in case_rows)
    weighted_cand_p95 = sum(r["cand_p95_us"] * r["weight"] for r in case_rows)

    weighted_p50_speedup = weighted_base_p50 / weighted_cand_p50
    weighted_p95_ratio = weighted_cand_p95 / weighted_base_p95

    aggregate_p50_ok = weighted_p50_speedup >= thresholds.min_weighted_p50_speedup
    aggregate_p95_ok = weighted_p95_ratio <= thresholds.max_weighted_p95_ratio
    all_cases_pass = all(r["verdict"] == "PASS" for r in case_rows)
    overall_pass = all_cases_pass and aggregate_p50_ok and aggregate_p95_ok

    return {
        "baseline_gpu": base_report["gpu_name"],
        "candidate_gpu": cand_report["gpu_name"],
        "thresholds": dataclasses.asdict(thresholds),
        "cases": case_rows,
        "aggregate": {
            "weighted_p50_speedup": weighted_p50_speedup,
            "weighted_p95_ratio": weighted_p95_ratio,
            "p50_ok": aggregate_p50_ok,
            "p95_ok": aggregate_p95_ok,
            "all_cases_pass": all_cases_pass,
            "overall_verdict": "PASS" if overall_pass else "FAIL",
        },
    }


def _print_ab_report(ab_report: dict[str, Any]) -> None:
    print(f"Benchmark suite: {ab_report.get('benchmark_suite_id', BENCHMARK_SUITE_ID)}")
    print(f"Model preset: {ab_report.get('model', '?')}")
    print(
        f"Baseline GPU: {ab_report['baseline_gpu']} | "
        f"Candidate GPU: {ab_report['candidate_gpu']}"
    )
    thresholds = ab_report["thresholds"]
    print(
        "Thresholds: "
        f"min_case_p50_speedup={thresholds['min_case_p50_speedup']:.3f}, "
        f"min_weighted_p50_speedup={thresholds['min_weighted_p50_speedup']:.3f}, "
        f"max_case_p95_ratio={thresholds['max_case_p95_ratio']:.3f}, "
        f"max_weighted_p95_ratio={thresholds['max_weighted_p95_ratio']:.3f}"
    )
    print("-" * 150)
    print(
        f"{'case':32} {'M':>5} {'N':>5} {'K':>5} {'base_p50':>10} "
        f"{'cand_p50':>10} {'p50_spd':>9} {'p95_ratio':>10} {'verdict':>8}"
    )
    for row in ab_report["cases"]:
        case = row.get("case") or {}
        shape = (
            f"{case.get('m', '?'):>5} {case.get('n', '?'):>5} "
            f"{case.get('k', '?'):>5}"
            if case
            else f"{'?':>5} {'?':>5} {'?':>5}"
        )
        print(
            f"{row['case_name']:32} {shape} "
            f"{row['base_p50_us']:10.2f} {row['cand_p50_us']:10.2f} "
            f"{row['p50_speedup']:9.3f} {row['p95_ratio']:10.3f} "
            f"{row['verdict']:>8}"
        )
        if row["verdict"] == "FAIL" and row["reason"]:
            print(f"  reason: {row['reason']}")

    agg = ab_report["aggregate"]
    print("-" * 150)
    print(
        f"Weighted p50 speedup: {agg['weighted_p50_speedup']:.3f}x | "
        f"Weighted p95 ratio: {agg['weighted_p95_ratio']:.3f}x"
    )
    print(f"Overall verdict: {agg['overall_verdict']}")


def _spawn_single_run(
    root: Path,
    script_path: Path,
    warmup: int,
    repeat: int,
    seed: int,
    timing: str,
    output_json: Path,
    case_names: list[str],
    model: str,
) -> None:
    env = os.environ.copy()
    old_pythonpath = env.get("PYTHONPATH", "")
    repo_path = str(root)
    env["PYTHONPATH"] = f"{repo_path}:{old_pythonpath}" if old_pythonpath else repo_path

    cmd = [
        sys.executable,
        str(script_path),
        "--mode",
        "single",
        "--warmup",
        str(warmup),
        "--repeat",
        str(repeat),
        "--seed",
        str(seed),
        "--timing",
        timing,
        "--model",
        model,
        "--output-json",
        str(output_json),
    ]
    if case_names:
        cmd.extend(["--cases", *case_names])

    print(f"[spawn] repo={root} script={script_path} model={model}")
    subprocess.run(cmd, env=env, check=True, cwd=str(root))


def _select_cases(case_names: list[str], model: str) -> list[MarlinMoECase]:
    all_cases = cases_for_model(model)
    if not case_names:
        return all_cases
    case_map = {c.name: c for c in all_cases}
    missing = [name for name in case_names if name not in case_map]
    if missing:
        raise ValueError(
            f"Unknown case names for model={model}: {missing}. "
            f"Available: {sorted(case_map)}"
        )
    return [case_map[name] for name in case_names]


def _thresholds_from_args(args: argparse.Namespace) -> ThresholdConfig:
    min_case_p50 = args.min_case_p50_speedup
    min_weighted_p50 = args.min_weighted_p50_speedup
    if args.require_improvement:
        if min_case_p50 == 1.0:
            min_case_p50 = 1.02
        if min_weighted_p50 == 1.0:
            min_weighted_p50 = 1.02
    return ThresholdConfig(
        min_case_p50_speedup=min_case_p50,
        min_weighted_p50_speedup=min_weighted_p50,
        max_case_p95_ratio=args.max_case_p95_ratio,
        max_weighted_p95_ratio=args.max_weighted_p95_ratio,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM Marlin MoE and compare baseline/candidate."
    )
    parser.add_argument("--mode", choices=["single", "ab"], default="single")
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="Print the selected case matrix and exit (no GPU run).",
    )
    parser.add_argument(
        "--model",
        choices=["glm5", "glm5_tp1", "glm5_ep8", "glm5_full", "generic"],
        default="glm5",
        help=(
            "Built-in workload preset. Default glm5 is GLM-5 with TP=8 "
            "(K=6144, N=256, E=256, topk=8)."
        ),
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--timing",
        choices=["event", "batch"],
        default="event",
        help=(
            "event: per-repeat CUDA events (mean/p50/p95/std). "
            "batch: one event around all repeats (average latency)."
        ),
    )
    parser.add_argument("--cases", nargs="*", default=[])
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--output-csv", type=str, default="")
    parser.add_argument("--baseline-root", type=str, default="")
    parser.add_argument("--candidate-root", type=str, default="")
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--require-improvement", action="store_true")
    parser.add_argument("--min-case-p50-speedup", type=float, default=1.0)
    parser.add_argument("--min-weighted-p50-speedup", type=float, default=1.0)
    parser.add_argument("--max-case-p95-ratio", type=float, default=1.05)
    parser.add_argument("--max-weighted-p95-ratio", type=float, default=1.05)
    args = parser.parse_args()

    cases = _select_cases(args.cases, model=args.model)
    _validate_case_selection(args.model, cases)

    if args.list_cases:
        _print_case_plan(args.model, cases)
        return

    if args.mode == "single":
        _print_case_plan(args.model, cases)
        report = _run_single(
            cases,
            warmup=args.warmup,
            repeat=args.repeat,
            seed=args.seed,
            timing=args.timing,
        )
        report["model"] = args.model
        report["benchmark_suite_id"] = BENCHMARK_SUITE_ID
        report["model_description"] = _describe_preset(args.model)
        _print_single_report(report)
        if args.output_json:
            _write_json(Path(args.output_json), report)
        if args.output_csv:
            csv_rows = []
            for row in report["results"]:
                c = row["case"]
                csv_rows.append(
                    {
                        "case": c["name"],
                        "m": c["m"],
                        "n": c["n"],
                        "k": c["k"],
                        "e": c["e"],
                        "topk": c["topk"],
                        "group_size": c["group_size"],
                        "act_order": c["act_order"],
                        "is_k_full": c["is_k_full"],
                        "dtype": c["dtype"],
                        "weight": c.get("weight"),
                        "timing": row.get("timing", args.timing),
                        "mean_us": row["mean_us"],
                        "p50_us": row["p50_us"],
                        "p95_us": row["p95_us"],
                        "std_us": row["std_us"],
                        "min_us": row["min_us"],
                        "max_us": row["max_us"],
                        "total_us": row.get("total_us"),
                    }
                )
            _write_csv(Path(args.output_csv), csv_rows)
        return

    if not args.baseline_root or not args.candidate_root:
        raise ValueError("--baseline-root and --candidate-root are required in --mode ab")

    baseline_root = Path(args.baseline_root).resolve()
    candidate_root = Path(args.candidate_root).resolve()
    baseline_script = _resolve_benchmark_script(baseline_root)
    candidate_script = _resolve_benchmark_script(candidate_root)
    thresholds = _thresholds_from_args(args)
    explicit_cases = bool(args.cases)
    selected_case_names = [c.name for c in cases] if explicit_cases else []

    print(f"Benchmark suite: {BENCHMARK_SUITE_ID}")
    print(f"Model preset: {args.model} ({_describe_preset(args.model)})")
    _print_case_plan(args.model, cases)
    print(f"Baseline script: {baseline_script}")
    print(f"Candidate script: {candidate_script}")

    with tempfile.TemporaryDirectory(prefix="marlin_moe_ab_") as td:
        base_json = Path(td) / "baseline.json"
        cand_json = Path(td) / "candidate.json"

        _spawn_single_run(
            root=baseline_root,
            script_path=baseline_script,
            warmup=args.warmup,
            repeat=args.repeat,
            seed=args.seed,
            timing=args.timing,
            output_json=base_json,
            case_names=selected_case_names,
            model=args.model,
        )
        _spawn_single_run(
            root=candidate_root,
            script_path=candidate_script,
            warmup=args.warmup,
            repeat=args.repeat,
            seed=args.seed,
            timing=args.timing,
            output_json=cand_json,
            case_names=selected_case_names,
            model=args.model,
        )
        base_report = _load_json(base_json)
        cand_report = _load_json(cand_json)
        ab_report = _build_ab_report(base_report, cand_report, thresholds)
        ab_report["benchmark_suite_id"] = BENCHMARK_SUITE_ID
        ab_report["model"] = args.model
        _print_ab_report(ab_report)

        if args.output_json:
            _write_json(
                Path(args.output_json),
                {
                    "model": args.model,
                    "baseline": base_report,
                    "candidate": cand_report,
                    "comparison": ab_report,
                },
            )
        if args.output_csv:
            _write_csv(Path(args.output_csv), ab_report["cases"])

        if args.gate and ab_report["aggregate"]["overall_verdict"] == "FAIL":
            sys.exit(1)


if __name__ == "__main__":
    main()
