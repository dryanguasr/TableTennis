"""Deterministic, resumable calibration for every versioned preset family.

The direct-service pass is deliberately small: service position and spin are
part of the definition, so only the three launch-velocity components are
fitted.  Higher-level racket, return, and exercise passes consume the promoted
direct/service presets instead of silently mixing old and new physics.
"""

from __future__ import annotations

import argparse
import json
import os
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from ..benchmarks.direct import (
    DEPTHS,
    LANES,
    MAX_FLIGHT_HEIGHT_ABOVE_NET_MM,
    MAX_REBOUND_HEIGHT_ABOVE_NET_MM,
    NET_CLEARANCE_LEVEL,
    SERVICE_TYPES,
    TARGET_MARGIN_MM,
    VELOCITY_OVERRIDES,
)
from ..constants import (
    BALL_RADIUS,
    NET_HEIGHT,
    TABLE_HEIGHT,
    TABLE_LENGTH,
    TABLE_WIDTH,
)
from ..events import net_crossings, table_bounces
from ..models import InitialConditions
from ..physics import simulate
from ..physics import simulate_racket_impact
from ..presets.returns import (
    PILOT_SERVICE_PARAMS,
    PROFILE_TARGETS,
    PROFILE_TARGET_X,
)


@dataclass(frozen=True)
class DirectRetuneJob:
    """Serializable definition of one direct-service calibration."""

    service: str
    depth: str
    lane: str
    maxiter: int = 18

    @property
    def key(self) -> str:
        return f"{self.service}/{self.depth}/{self.lane}"


@dataclass(frozen=True)
class ReturnRetuneJob:
    profile: str
    maxiter: int = 70
    popsize: int = 6
    restarts: int = 2


@dataclass(frozen=True)
class ServiceRetuneJob:
    name: str
    depth: str
    direction: str
    maxiter: int = 55


def retune_service_job(job: ServiceRetuneJob) -> dict[str, object]:
    from ..exchange import ServiceTargets
    from ..search.returns import ReturnSearchConfig, search_service

    targets = ServiceTargets(
        depth=job.depth,
        direction=job.direction,
        spin_rps=(0.0, -45.0, 35.0),
        bounce_tolerance_mm=100.0 if job.depth == "long" else 75.0,
        spin_tolerance_rps=12.0 if job.depth == "long" else 10.0,
        max_height_above_net_mm=50.0,
        max_rebound_height_above_net_mm=25.0,
    )
    result = search_service(
        targets,
        config=ReturnSearchConfig(
            maxiter=job.maxiter,
            popsize=6,
            restarts=2,
            workers=1,
            seed=20260704
            + ("pilot_short", "long_backhand", "long_forehand").index(job.name)
            * 1009,
        ),
    )
    return {
        "key": job.name,
        "name": job.name,
        "passed": result.success,
        "cost": result.cost,
        "vector": list(result.vector),
        "params": asdict(result.params),
        "validation": result.validation.to_dict(),
        "optimizer": result.optimizer,
        "message": result.message,
    }


def retune_return_job(job: ReturnRetuneJob) -> dict[str, object]:
    """Search one shared forehand/backhand return vector."""

    from scipy.optimize import differential_evolution

    from ..exchange import (
        ContactSelection,
        RubberProperties,
        StrokeTargets,
        contact_state,
    )
    from ..search.returns import racket_parameters_for_target_spin
    from ..validation import validate_return

    depth, spin = PROFILE_TARGETS[job.profile]
    service_result = simulate_racket_impact(PILOT_SERVICE_PARAMS, t_max=3.0)
    targets = StrokeTargets(
        depth=depth,
        direction="elbow",
        spin_rps=spin,
        stroke_side="forehand",
        target_x=PROFILE_TARGET_X.get(job.profile),
    )
    rubber = RubberProperties()
    cut_stroke = spin[1] > 0
    fraction_bounds = (0.50, 0.85) if cut_stroke else (0.0, 1.0)

    def candidate(values):
        selection = ContactSelection(moment=4, fraction=float(values[0]))
        index, incoming = contact_state(service_result, selection)
        params = racket_parameters_for_target_spin(
            incoming,
            values[3:6],
            float(values[1]),
            float(values[2]),
            rubber,
            "forehand",
        )
        result = simulate_racket_impact(params, t_max=3.0)
        report = validate_return(params, result, targets)
        return selection, index, params, result, report

    def score(values):
        _, _, params, result, report = candidate(values)
        value = (report.bounce_error_mm / targets.bounce_tolerance_mm) ** 2
        value += (report.spin_error_rps / targets.spin_tolerance_rps) ** 2
        value += 2_000.0 * len(report.violations)
        if np.isfinite(report.net_clearance_mm):
            value += max(0.0, 5.0 - report.net_clearance_mm) ** 2
        else:
            value += 10_000.0
        if cut_stroke:
            value += max(0.0, params.racket_velocity[0] / 500.0) ** 2 * 100.0
            value += max(0.0, (params.racket_velocity[2] + 100.0) / 500.0) ** 2 * 100.0
        else:
            value += max(0.0, (100.0 - params.racket_velocity[2]) / 500.0) ** 2 * 100.0
        if targets.depth in {"short", "two_bounce"} and report.bounces_on_target_side < 2:
            bounces = table_bounces(result, "server")
            if bounces:
                first = bounces[0]
                index = first.index
                flight_time = max(0.0, 2.0 * result.v[2, index] / 9810.0)
                predicted_x = first.point[0] + result.v[0, index] * flight_time
                value += max(0.0, BALL_RADIUS - predicted_x) ** 2 / 25.0
        return float(value)

    seed = 20260704 + tuple(PROFILE_TARGETS).index(job.profile) * 1009
    spin_bounds = tuple(
        (
            float(component - targets.spin_tolerance_rps),
            float(component + targets.spin_tolerance_rps),
        )
        for component in targets.spin_rps
    )
    best = None
    for restart in range(job.restarts):
        fit = differential_evolution(
            score,
            bounds=(
                fraction_bounds,
                (0.0, 2 * np.pi),
                (-24_000.0, 24_000.0),
                *spin_bounds,
            ),
            maxiter=job.maxiter,
            popsize=job.popsize,
            seed=seed + restart * 1009,
            polish=True,
            workers=1,
            tol=0.002,
        )
        selection, index, params, _, validation = candidate(fit.x)
        current = (float(fit.fun), selection, index, params, validation)
        if best is None or current[0] < best[0]:
            best = current
        if validation.passed:
            break
    assert best is not None
    cost, selection, index, params, validation = best
    vector = [
        selection.fraction,
        params.racket_angle[1],
        params.racket_angle[2],
        *params.racket_velocity,
    ]
    return {
        "key": job.profile,
        "profile": job.profile,
        "passed": validation.passed,
        "cost": cost,
        "vector": [float(value) for value in vector],
        "validation": validation.to_dict(),
        "optimizer": "spin_inversion+differential_evolution",
        "message": "Searched contact phase, racket-normal phase, and normal impulse.",
    }


def _direct_metrics(job: DirectRetuneJob, velocity: Iterable[float]) -> dict[str, object]:
    service = SERVICE_TYPES[job.service]
    result = simulate(
        InitialConditions(
            pos=service["position"],
            vel=tuple(float(value) for value in velocity),
            omega=service["omega"],
        ),
        dt=0.005,
        t_max=1.8,
    )
    bounces = table_bounces(result)
    receiver = [
        event
        for event in table_bounces(result, "receiver")
        if event.point[0] < TABLE_LENGTH - BALL_RADIUS
        and BALL_RADIUS < event.point[1] < TABLE_WIDTH - BALL_RADIUS
    ]
    target = np.array(
        (DEPTHS[job.depth]["target_x"], LANES[job.lane]["y"]),
        dtype=float,
    )
    landing_proxy = None
    if receiver:
        landing_proxy = tuple(float(value) for value in receiver[0].point[:2])
    elif bounces:
        start = bounces[0].index + 1
        table_level = TABLE_HEIGHT + BALL_RADIUS
        for index in range(start + 1, result.x.shape[1]):
            if (
                result.x[0, index] >= TABLE_LENGTH / 2
                and result.x[2, index - 1] >= table_level
                and result.x[2, index] < table_level
            ):
                z0 = float(result.x[2, index - 1])
                z1 = float(result.x[2, index])
                fraction = (table_level - z0) / (z1 - z0)
                point = result.x[:, index - 1] + fraction * (
                    result.x[:, index] - result.x[:, index - 1]
                )
                landing_proxy = (float(point[0]), float(point[1]))
                break
    target_error = (
        float(np.linalg.norm(np.asarray(landing_proxy) - target))
        if landing_proxy is not None
        else float("inf")
    )
    ordered = (
        len(bounces) >= 2
        and bounces[0].side == "server"
        and bounces[1].side == "receiver"
    )
    cutoff = bounces[1].time if len(bounces) >= 2 else float("inf")
    touches_net = any(
        event.kind == "net_contact" and event.time <= cutoff
        for event in result.events
    )
    crossings = net_crossings(result)
    clearance = (
        float(
            crossings[0].point[2] - NET_CLEARANCE_LEVEL
        )
        if crossings
        else float("-inf")
    )
    if ordered:
        flight_height = float(
            np.max(result.x[2, bounces[0].index : bounces[1].index + 1])
            - NET_CLEARANCE_LEVEL
        )
        later_bounces = [
            event.index for event in bounces if event.index > bounces[1].index
        ]
        rebound_end = (
            later_bounces[0] if later_bounces else result.x.shape[1] - 1
        )
        rebound_height = float(
            np.max(result.x[2, bounces[1].index : rebound_end + 1])
            - NET_CLEARANCE_LEVEL
        )
    else:
        flight_height = float("inf")
        rebound_height = float("inf")
    target_count = len(receiver)
    depth_valid = (
        target_count >= 2
        if job.depth in {"short", "two_bounce"}
        else target_count == 1
    )
    passed = bool(
        ordered
        and not touches_net
        and clearance >= 5.0
        and target_error <= TARGET_MARGIN_MM
        and depth_valid
        and flight_height <= MAX_FLIGHT_HEIGHT_ABOVE_NET_MM
        and rebound_height <= MAX_REBOUND_HEIGHT_ABOVE_NET_MM
    )
    server_point = (
        tuple(float(value) for value in bounces[0].point[:2])
        if bounces and bounces[0].side == "server"
        else None
    )
    receiver_point = (
        tuple(float(value) for value in receiver[0].point[:2])
        if receiver
        else None
    )
    return {
        "passed": passed,
        "target_error_mm": target_error,
        "net_clearance_mm": clearance,
        "max_height_above_net_mm": flight_height,
        "rebound_height_above_net_mm": rebound_height,
        "server_bounce": server_point,
        "receiver_bounce": receiver_point,
        "landing_proxy": landing_proxy,
        "receiver_bounces": target_count,
        "ordered_bounces": ordered,
        "touches_net": touches_net,
    }


def _direct_score(job: DirectRetuneJob, velocity: np.ndarray) -> float:
    metrics = _direct_metrics(job, velocity)
    target_error = float(metrics["target_error_mm"])
    score = target_error if np.isfinite(target_error) else 4_000.0
    if not metrics["ordered_bounces"]:
        score += 5_000.0
    if metrics["touches_net"]:
        score += 4_000.0
    clearance = float(metrics["net_clearance_mm"])
    score += (
        max(0.0, 5.0 - clearance) * 80.0
        if np.isfinite(clearance)
        else 5_000.0
    )
    flight_height = float(metrics["max_height_above_net_mm"])
    rebound_height = float(metrics["rebound_height_above_net_mm"])
    score += max(0.0, flight_height - MAX_FLIGHT_HEIGHT_ABOVE_NET_MM) ** 2 * 4.0
    score += max(0.0, rebound_height - MAX_REBOUND_HEIGHT_ABOVE_NET_MM) ** 2 * 6.0
    server_bounce = metrics["server_bounce"]
    server_x = float(server_bounce[0]) if server_bounce is not None else -500.0
    # Do not let the optimizer balance on the mathematical table edge.
    score += max(0.0, BALL_RADIUS - server_x) * 80.0
    receiver_bounces = int(metrics["receiver_bounces"])
    if job.depth in {"short", "two_bounce"} and receiver_bounces < 2:
        score += 4_000.0
    if job.depth == "long" and receiver_bounces != 1:
        score += 4_000.0
    return float(score)


def retune_direct_job(job: DirectRetuneJob) -> dict[str, object]:
    """Fit and validate one job; safe to send to a Windows process worker."""

    from scipy.optimize import differential_evolution

    seed = zlib.crc32(job.key.encode("utf-8")) & 0xFFFFFFFF
    fit = differential_evolution(
        lambda vector: _direct_score(job, vector),
        bounds=((3500.0, 10500.0), (-6500.0, 6500.0), (-1400.0, -50.0)),
        seed=seed,
        popsize=6,
        maxiter=job.maxiter,
        tol=0.002,
        polish=True,
        workers=1,
        updating="immediate",
    )
    velocity = fit.x
    metrics = _direct_metrics(job, velocity)
    return {
        "key": job.key,
        "service": job.service,
        "depth": job.depth,
        "lane": job.lane,
        "velocity": [round(float(value), 4) for value in velocity],
        **metrics,
    }


def direct_jobs(maxiter: int = 18) -> list[DirectRetuneJob]:
    return [
        DirectRetuneJob(service, depth, lane, maxiter=maxiter)
        for service in SERVICE_TYPES
        for depth in DEPTHS
        for lane in LANES
    ]


def _load_results(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(row["key"]): row
        for row in payload.get("results", [])
        if isinstance(row, dict) and "key" in row
    }


def _write_results(path: Path, rows: dict[str, dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = [rows[key] for key in sorted(rows)]
    payload = {
        "model": "ACE baseline",
        "complete": len(ordered) == 54 and all(row.get("passed") for row in ordered),
        "results": ordered,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run_direct_retune(
    output: Path,
    workers: int = 1,
    maxiter: int = 18,
    overwrite: bool = False,
) -> dict[str, dict[str, object]]:
    """Run or resume all direct jobs, checkpointing after every completion."""

    rows = {} if overwrite else _load_results(output)
    # A checkpoint is reusable only if it still validates against the current
    # service definition. This matters when a failed family needs its spin
    # definition adjusted between deterministic passes.
    for key, row in tuple(rows.items()):
        if not row.get("passed"):
            continue
        job = DirectRetuneJob(
            str(row["service"]),
            str(row["depth"]),
            str(row["lane"]),
            maxiter=maxiter,
        )
        metrics = _direct_metrics(job, row["velocity"])
        if not metrics["passed"]:
            rows[key] = {**row, **metrics}
    jobs = [
        job
        for job in direct_jobs(maxiter=maxiter)
        if not rows.get(job.key, {}).get("passed")
    ]
    total = len(direct_jobs(maxiter=maxiter))
    completed = total - len(jobs)
    if workers == 1:
        for job in jobs:
            row = retune_direct_job(job)
            rows[job.key] = row
            completed += 1
            _write_results(output, rows)
            print(
                f"[{completed}/{total}] {job.key}: "
                f"{'OK' if row['passed'] else 'FAIL'} "
                f"error={row['target_error_mm']:.1f} mm "
                f"height={row['max_height_above_net_mm']:.1f} mm "
                f"rebound={row['rebound_height_above_net_mm']:.1f} mm"
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(retune_direct_job, job): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                row = future.result()
                rows[job.key] = row
                completed += 1
                _write_results(output, rows)
                print(
                    f"[{completed}/{total}] {job.key}: "
                    f"{'OK' if row['passed'] else 'FAIL'} "
                    f"error={row['target_error_mm']:.1f} mm "
                    f"height={row['max_height_above_net_mm']:.1f} mm "
                    f"rebound={row['rebound_height_above_net_mm']:.1f} mm"
                )
    _write_results(output, rows)
    return rows


def run_return_retune(
    output: Path,
    workers: int = 1,
    maxiter: int = 70,
    overwrite: bool = False,
) -> dict[str, dict[str, object]]:
    rows = {} if overwrite else _load_results(output)
    jobs = [
        ReturnRetuneJob(profile, maxiter=maxiter)
        for profile in PROFILE_TARGETS
        if not rows.get(profile, {}).get("passed")
    ]
    total = len(PROFILE_TARGETS)
    completed = total - len(jobs)
    if workers == 1:
        iterator = ((job, retune_return_job(job)) for job in jobs)
        for job, row in iterator:
            rows[job.profile] = row
            completed += 1
            _write_results(output, rows)
            print(
                f"[{completed}/{total}] return/{job.profile}: "
                f"{'OK' if row['passed'] else 'FAIL'}"
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(retune_return_job, job): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                row = future.result()
                rows[job.profile] = row
                completed += 1
                _write_results(output, rows)
                print(
                    f"[{completed}/{total}] return/{job.profile}: "
                    f"{'OK' if row['passed'] else 'FAIL'}"
                )
    # Return files contain five profiles rather than the direct suite's 54.
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": "ACE baseline",
        "complete": len(rows) == total and all(row.get("passed") for row in rows.values()),
        "results": [rows[key] for key in sorted(rows)],
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return rows


def _run_jobs(
    jobs,
    worker,
    output: Path,
    workers: int,
    overwrite: bool,
    label: str,
) -> dict[str, dict[str, object]]:
    rows = {} if overwrite else _load_results(output)
    pending = [job for job in jobs if not rows.get(job.name, {}).get("passed")]
    completed = len(jobs) - len(pending)
    if workers == 1:
        completed_rows = ((job, worker(job)) for job in pending)
        for job, row in completed_rows:
            rows[job.name] = row
            completed += 1
            _write_generic_results(output, rows, len(jobs))
            print(
                f"[{completed}/{len(jobs)}] {label}/{job.name}: "
                f"{'OK' if row['passed'] else 'FAIL'}"
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(worker, job): job for job in pending}
            for future in as_completed(futures):
                job = futures[future]
                row = future.result()
                rows[job.name] = row
                completed += 1
                _write_generic_results(output, rows, len(jobs))
                print(
                    f"[{completed}/{len(jobs)}] {label}/{job.name}: "
                    f"{'OK' if row['passed'] else 'FAIL'}"
                )
    _write_generic_results(output, rows, len(jobs))
    return rows


def _write_generic_results(
    output: Path,
    rows: dict[str, dict[str, object]],
    expected: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": "ACE baseline",
        "complete": len(rows) == expected and all(row.get("passed") for row in rows.values()),
        "results": [rows[key] for key in sorted(rows)],
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)


def run_service_retune(
    output: Path,
    workers: int = 1,
    maxiter: int = 55,
    overwrite: bool = False,
) -> dict[str, dict[str, object]]:
    jobs = [
        ServiceRetuneJob("pilot_short", "short", "elbow", maxiter),
        ServiceRetuneJob("long_backhand", "long", "backhand", maxiter),
        ServiceRetuneJob("long_forehand", "long", "forehand", maxiter),
    ]
    return _run_jobs(
        jobs,
        retune_service_job,
        output,
        min(workers, len(jobs)),
        overwrite,
        "service",
    )


def run_exercise_retune(
    output: Path,
    workers: int = 1,
    overwrite: bool = False,
) -> dict[str, dict[str, object]]:
    from .exercises import ExerciseSearchJob, search_exercise
    from ..presets.exercises import EXERCISE_NAMES

    jobs = [ExerciseSearchJob(name, 3) for name in EXERCISE_NAMES]
    rows = {} if overwrite else _load_results(output)
    pending = [job for job in jobs if not rows.get(job.name, {}).get("passed")]
    completed = len(jobs) - len(pending)
    if workers == 1:
        iterator = ((job, search_exercise(job)) for job in pending)
        for job, result in iterator:
            row = {"key": job.name, "name": job.name, **result}
            rows[job.name] = row
            completed += 1
            _write_generic_results(output, rows, len(jobs))
            print(f"[{completed}/{len(jobs)}] exercise/{job.name}: {'OK' if row['passed'] else 'FAIL'}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(search_exercise, job): job for job in pending}
            for future in as_completed(futures):
                job = futures[future]
                row = {"key": job.name, "name": job.name, **future.result()}
                rows[job.name] = row
                completed += 1
                _write_generic_results(output, rows, len(jobs))
                print(f"[{completed}/{len(jobs)}] exercise/{job.name}: {'OK' if row['passed'] else 'FAIL'}")
    _write_generic_results(output, rows, len(jobs))
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="table-tennis search retune-all",
        description="Recalibrate all versioned presets after a physics change.",
    )
    parser.add_argument(
        "--suite",
        choices=("direct", "services", "returns", "exercises", "all"),
        default="all",
        help="Calibration stage to run. 'all' starts with the dependency root.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--maxiter", type=int, default=18)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/search/retune"),
    )
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    jobs = direct_jobs(maxiter=args.maxiter)
    if args.dry_run:
        if args.suite in {"direct", "all"}:
            for index, job in enumerate(jobs, 1):
                print(f"[{index}/{len(jobs)}] DIRECT {job.key}")
        if args.suite in {"services", "all"}:
            for name in ("pilot_short", "long_backhand", "long_forehand"):
                print(f"SERVICE {name}")
        if args.suite in {"returns", "all"}:
            for index, profile in enumerate(PROFILE_TARGETS, 1):
                print(f"[{index}/{len(PROFILE_TARGETS)}] RETURN {profile}")
        if args.suite in {"exercises", "all"}:
            from ..presets.exercises import EXERCISE_NAMES

            for index, name in enumerate(EXERCISE_NAMES, 1):
                print(f"[{index}/{len(EXERCISE_NAMES)}] EXERCISE {name}")
        return
    if args.suite in {"direct", "all"}:
        output = args.output_dir / "direct_services.json"
        rows = run_direct_retune(
            output,
            workers=args.workers,
            maxiter=args.maxiter,
            overwrite=args.overwrite,
        )
        failed = [key for key, row in rows.items() if not row.get("passed")]
        if failed:
            raise SystemExit(
                f"Direct calibration incomplete: {len(failed)} failed. "
                f"Best candidates were preserved in {output}."
            )
    if args.suite in {"services", "all"}:
        output = args.output_dir / "services.json"
        rows = run_service_retune(
            output,
            workers=args.workers,
            maxiter=max(args.maxiter, 40),
            overwrite=args.overwrite,
        )
        failed = [key for key, row in rows.items() if not row.get("passed")]
        if failed:
            raise SystemExit(
                f"Service calibration incomplete: {len(failed)} failed. "
                f"Best candidates were preserved in {output}."
            )
    if args.suite in {"returns", "all"}:
        output = args.output_dir / "returns.json"
        rows = run_return_retune(
            output,
            workers=min(args.workers, len(PROFILE_TARGETS)),
            maxiter=max(args.maxiter, 50),
            overwrite=args.overwrite,
        )
        failed = [key for key, row in rows.items() if not row.get("passed")]
        if failed:
            raise SystemExit(
                f"Return calibration incomplete: {len(failed)} failed. "
                f"Best candidates were preserved in {output}."
            )
    if args.suite in {"exercises", "all"}:
        output = args.output_dir / "exercises.json"
        rows = run_exercise_retune(
            output,
            workers=args.workers,
            overwrite=args.overwrite,
        )
        failed = [key for key, row in rows.items() if not row.get("passed")]
        if failed:
            raise SystemExit(
                f"Exercise calibration incomplete: {len(failed)} failed. "
                f"Best candidates were preserved in {output}."
            )


if __name__ == "__main__":
    main()
