"""Resumable bulk MP4 generation for all benchmark suites."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from . import direct, racket
from ..physics import simulate, simulate_racket_impact
from ..presets.returns import PROFILE_TARGETS
from ..visualization import (
    animate_racket_impact,
    animate_simulation,
    resolve_ffmpeg_path,
)
from ..visualization.return_videos import save_exchange_video


SuiteName = Literal["direct", "racket", "returns"]
DEFAULT_OUTPUTS = {
    "direct": Path("outputs/benchmarks/direct"),
    "racket": Path("outputs/benchmarks/racket"),
    "returns": Path("outputs/benchmarks/returns"),
}


@dataclass(frozen=True)
class VideoJob:
    suite: SuiteName
    name: str
    output_path: str
    fps: int
    duration: float
    ffmpeg_path: str
    direct_case: direct.BenchmarkCase | None = None
    racket_case: racket.BenchmarkCase | None = None
    profile: str | None = None
    stroke_side: str | None = None


@dataclass(frozen=True)
class VideoJobResult:
    suite: str
    name: str
    path: str
    status: str
    duration: float | None = None
    error: str = ""


def resolve_ffprobe_path(ffmpeg_path: str | None = None) -> str | None:
    """Resolve FFprobe from PATH or next to an explicit FFmpeg executable."""

    if ffmpeg_path:
        ffmpeg = Path(ffmpeg_path)
        candidate = ffmpeg.with_name(
            "ffprobe.exe" if ffmpeg.suffix.lower() == ".exe" else "ffprobe"
        )
        if candidate.exists():
            return str(candidate)
    return shutil.which("ffprobe")


def probe_mp4_duration(
    path: Path | str,
    ffprobe_path: str | None = None,
) -> float | None:
    """Return MP4 duration when FFprobe can decode the file."""

    video = Path(path)
    if not video.is_file() or video.stat().st_size == 0:
        return None
    ffprobe = ffprobe_path or resolve_ffprobe_path()
    if ffprobe is None:
        return None
    try:
        completed = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        duration = float(completed.stdout.strip())
        return duration if duration > 0 else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def build_video_jobs(
    suites: tuple[SuiteName, ...] = ("direct", "racket", "returns"),
    *,
    fps: int = 30,
    duration: float = 5.0,
    ffmpeg_path: str,
) -> list[VideoJob]:
    """Build deterministic jobs for the selected benchmark suites."""

    jobs: list[VideoJob] = []
    if "direct" in suites:
        for case in direct.build_cases():
            jobs.append(
                VideoJob(
                    suite="direct",
                    name=f"{case.service}/{case.depth}/{case.lane}",
                    output_path=str(
                        DEFAULT_OUTPUTS["direct"] / direct.case_filename(case)
                    ),
                    fps=fps,
                    duration=duration,
                    ffmpeg_path=ffmpeg_path,
                    direct_case=case,
                )
            )
    if "racket" in suites:
        for case in racket.build_cases():
            jobs.append(
                VideoJob(
                    suite="racket",
                    name=f"{case.service}/{case.depth}/{case.lane}",
                    output_path=str(
                        DEFAULT_OUTPUTS["racket"] / racket.case_filename(case)
                    ),
                    fps=fps,
                    duration=duration,
                    ffmpeg_path=ffmpeg_path,
                    racket_case=case,
                )
            )
    if "returns" in suites:
        for profile in PROFILE_TARGETS:
            for stroke_side in ("forehand", "backhand"):
                jobs.append(
                    VideoJob(
                        suite="returns",
                        name=f"{profile}/{stroke_side}",
                        output_path=str(
                            DEFAULT_OUTPUTS["returns"]
                            / f"{profile}_{stroke_side}.mp4"
                        ),
                        fps=fps,
                        duration=duration,
                        ffmpeg_path=ffmpeg_path,
                        profile=profile,
                        stroke_side=stroke_side,
                    )
                )
    return jobs


def render_video_job(job: VideoJob) -> VideoJobResult:
    """Render one job to a temporary file and atomically publish it."""

    output = Path(job.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.stem}.tmp{output.suffix}")
    if temporary.exists():
        temporary.unlink()
    try:
        if job.suite == "direct" and job.direct_case is not None:
            result = simulate(
                job.direct_case.initial_conditions,
                t_max=job.duration,
            )
            animate_simulation(
                result,
                save=str(temporary),
                ffmpeg_path=job.ffmpeg_path,
                fps=job.fps,
            )
        elif job.suite == "racket" and job.racket_case is not None:
            result = simulate_racket_impact(
                job.racket_case.params,
                t_max=job.duration,
            )
            animate_racket_impact(
                result,
                job.racket_case.params,
                save=str(temporary),
                ffmpeg_path=job.ffmpeg_path,
                fps=job.fps,
            )
        elif (
            job.suite == "returns"
            and job.profile is not None
            and job.stroke_side is not None
        ):
            save_exchange_video(
                job.profile,
                job.stroke_side,
                temporary,
                job.ffmpeg_path,
                fps=job.fps,
                max_duration=job.duration,
            )
        else:
            raise ValueError(f"Incomplete video job: {job}")

        duration = probe_mp4_duration(
            temporary,
            resolve_ffprobe_path(job.ffmpeg_path),
        )
        if duration is None:
            raise RuntimeError("FFprobe could not validate the rendered MP4.")
        temporary.replace(output)
        return VideoJobResult(
            suite=job.suite,
            name=job.name,
            path=str(output),
            status="rendered",
            duration=duration,
        )
    except Exception as exc:
        if temporary.exists():
            temporary.unlink()
        return VideoJobResult(
            suite=job.suite,
            name=job.name,
            path=str(output),
            status="failed",
            error=str(exc),
        )


def _parse_suites(values: list[str] | None) -> tuple[SuiteName, ...]:
    if not values or "all" in values:
        return ("direct", "racket", "returns")
    return tuple(dict.fromkeys(values))  # type: ignore[return-value]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="table-tennis generate benchmark-videos",
        description="Generate resumable benchmark MP4 batches.",
    )
    parser.add_argument(
        "--suite",
        action="append",
        choices=["direct", "racket", "returns", "all"],
        help="Suite to render; repeat the option or use all.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--ffmpeg")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/benchmarks/video_manifest.json"),
    )
    args = parser.parse_args(argv)

    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.fps < 1 or args.duration <= 0:
        parser.error("--fps and --duration must be positive")

    ffmpeg = resolve_ffmpeg_path(args.ffmpeg)
    if ffmpeg is None:
        parser.error(
            "MP4 output requires FFmpeg. Install it or pass --ffmpeg."
        )
    ffprobe = resolve_ffprobe_path(ffmpeg)
    if ffprobe is None:
        parser.error(
            "Resumable video validation requires FFprobe next to FFmpeg or in PATH."
        )

    jobs = build_video_jobs(
        _parse_suites(args.suite),
        fps=args.fps,
        duration=args.duration,
        ffmpeg_path=ffmpeg,
    )
    if args.limit is not None:
        jobs = jobs[: max(0, args.limit)]

    results: list[VideoJobResult] = []
    pending: list[VideoJob] = []
    total = len(jobs)
    for index, job in enumerate(jobs, start=1):
        existing_duration = probe_mp4_duration(job.output_path, ffprobe)
        if existing_duration is not None and not args.overwrite:
            result = VideoJobResult(
                suite=job.suite,
                name=job.name,
                path=job.output_path,
                status="skipped",
                duration=existing_duration,
            )
            results.append(result)
            print(f"[{index}/{total}] SKIP {job.output_path}")
        elif args.dry_run:
            status = "OVERWRITE" if existing_duration is not None else "RENDER"
            results.append(
                VideoJobResult(
                    suite=job.suite,
                    name=job.name,
                    path=job.output_path,
                    status="dry-run",
                    duration=existing_duration,
                )
            )
            print(f"[{index}/{total}] {status} {job.output_path}")
        else:
            pending.append(job)

    if pending:
        completed_count = total - len(pending)
        if args.workers == 1:
            for job in pending:
                result = render_video_job(job)
                completed_count += 1
                results.append(result)
                print(
                    f"[{completed_count}/{total}] {result.status.upper()} "
                    f"{result.path}"
                )
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(render_video_job, job): job
                    for job in pending
                }
                for future in as_completed(futures):
                    result = future.result()
                    completed_count += 1
                    results.append(result)
                    print(
                        f"[{completed_count}/{total}] {result.status.upper()} "
                        f"{result.path}"
                    )

    results.sort(key=lambda item: (item.suite, item.name))
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps([asdict(result) for result in results], indent=2),
        encoding="utf-8",
    )
    failures = [result for result in results if result.status == "failed"]
    counts = {
        status: sum(result.status == status for result in results)
        for status in ("rendered", "skipped", "dry-run", "failed")
    }
    print(
        "summary "
        + " ".join(f"{key}={value}" for key, value in counts.items())
        + f" manifest={args.manifest}"
    )
    if failures:
        for failure in failures:
            print(f"ERROR {failure.path}: {failure.error}")
        raise SystemExit(1)
