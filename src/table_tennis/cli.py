"""Unified command-line interface for the table-tennis simulator."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from .benchmarks import direct, racket, returns, videos
from .doctor import run_doctor
from .models import InitialConditions
from .physics import simulate
from .search import exercises, retune_all, service
from .visualization import animate_simulation
from .visualization import exercise_viewer, exercise_videos, return_videos, viewer


COMMANDS = """\
Commands:
  doctor
  simulate
  benchmark direct
  benchmark racket
  benchmark returns
  search service
  search exercise
  search retune-all
  generate return-videos
  generate benchmark-videos
  generate exercise-videos
  generate exercise-viewer
  generate racket-viewer

Run `table-tennis <command> --help` for command-specific options.
"""


def _simulate_main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="table-tennis simulate",
        description="Simulate and display or save the default trajectory.",
    )
    parser.add_argument("--save", metavar="FILE", help="Save an MP4 animation.")
    parser.add_argument("--ffmpeg", help="Path or executable name for FFmpeg.")
    args = parser.parse_args(argv)
    result = simulate(InitialConditions())
    animate_simulation(
        result,
        save=args.save,
        ffmpeg_path=args.ffmpeg,
    )


def _group_help(group: str, names: tuple[str, ...]) -> None:
    choices = "\n".join(f"  {group} {name}" for name in names)
    print(f"Available {group} commands:\n{choices}")


def _dispatch_group(
    group: str,
    argv: list[str],
    commands: dict[str, Callable[[list[str] | None], None]],
) -> None:
    if not argv or argv[0] in {"-h", "--help"}:
        _group_help(group, tuple(commands))
        return
    name = argv[0]
    if name not in commands:
        raise SystemExit(
            f"Unknown {group} command: {name}\n"
            f"Choose one of: {', '.join(commands)}"
        )
    commands[name](argv[1:])


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print("Table Tennis Physics Simulator\n")
        print(COMMANDS)
        return
    command = args.pop(0)
    if command == "doctor":
        if args:
            raise SystemExit("doctor does not accept arguments")
        if not run_doctor():
            raise SystemExit(1)
    elif command == "simulate":
        _simulate_main(args)
    elif command == "benchmark":
        _dispatch_group(
            "benchmark",
            args,
            {
                "direct": direct.main,
                "racket": racket.main,
                "returns": returns.main,
            },
        )
    elif command == "search":
        _dispatch_group(
            "search",
            args,
            {
                "service": service.main,
                "exercise": exercises.main,
                "retune-all": retune_all.main,
            },
        )
    elif command == "generate":
        _dispatch_group(
            "generate",
            args,
            {
                "return-videos": return_videos.main,
                "benchmark-videos": videos.main,
                "exercise-videos": exercise_videos.main,
                "exercise-viewer": exercise_viewer.main,
                "racket-viewer": viewer.main,
            },
        )
    else:
        raise SystemExit(f"Unknown command: {command}\n\n{COMMANDS}")
