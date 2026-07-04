"""Recalibrate exercise presets and export reproducible candidate parameters."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from ..physics import apply_racket_impact
from ..presets.exercises import EXERCISE_NAMES, build_exercise
from ..rally import simulate_exercise


@dataclass(frozen=True)
class ExerciseSearchJob:
    name: str
    cycles: int = 3


def search_exercise(job: ExerciseSearchJob) -> dict[str, object]:
    """Calibrate one complete exercise and return a JSON-safe candidate."""

    result = simulate_exercise(
        build_exercise(job.name, cycles=job.cycles),
        use_calibrated_preset=False,
    )
    strokes = []
    for index, segment in enumerate(result.segments, start=1):
        post = apply_racket_impact(segment.params)
        strokes.append(
            {
                "index": index,
                "label": segment.stroke.label,
                "cycle": segment.stroke.cycle,
                "stroke": asdict(segment.stroke),
                "racket_angle": list(segment.params.racket_angle),
                "racket_velocity": list(segment.params.racket_velocity),
                "post_velocity": list(post.vel),
                "post_spin_rps": [
                    float(value / (2.0 * 3.141592653589793))
                    for value in post.omega
                ],
                "validation": asdict(segment.validation),
            }
        )
    return {
        "exercise": job.name,
        "cycles": job.cycles,
        "passed": result.passed,
        "violations": list(result.violations),
        "duration": result.duration,
        "strokes": strokes,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="table-tennis search exercise",
        description=__doc__,
    )
    parser.add_argument(
        "--exercise",
        action="append",
        choices=EXERCISE_NAMES,
        help="Exercise to recalibrate; repeat it or omit it for all.",
    )
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/search/exercises/candidates.json"),
    )
    args = parser.parse_args(argv)
    if args.cycles < 3:
        parser.error("--cycles must be at least 3")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    names = args.exercise or list(EXERCISE_NAMES)
    jobs = [ExerciseSearchJob(name, args.cycles) for name in names]
    if args.workers == 1:
        candidates = []
        for index, job in enumerate(jobs, start=1):
            print(f"[{index}/{len(jobs)}] CALIBRATE {job.name}")
            candidates.append(search_exercise(job))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            candidates = list(executor.map(search_exercise, jobs))
        for index, candidate in enumerate(candidates, start=1):
            print(
                f"[{index}/{len(candidates)}] CALIBRATED "
                f"{candidate['exercise']}"
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(candidates, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    failures = [item for item in candidates if not item["passed"]]
    print(
        f"summary total={len(candidates)} failed={len(failures)} "
        f"output={args.output}"
    )
    if failures:
        raise SystemExit(1)
