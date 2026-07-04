"""Environment diagnostics for notebooks, searches, and MP4 generation."""

from __future__ import annotations

import importlib
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class DiagnosticCheck:
    name: str
    passed: bool
    detail: str


def _dependency_check(import_name: str, distribution: str) -> DiagnosticCheck:
    try:
        importlib.import_module(import_name)
        version = metadata.version(distribution)
        return DiagnosticCheck(distribution, True, version)
    except Exception as exc:
        return DiagnosticCheck(distribution, False, str(exc))


def _kernel_check() -> DiagnosticCheck:
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "jupyter", "kernelspec", "list", "--json"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        kernels = json.loads(completed.stdout).get("kernelspecs", {})
        resource_dir = kernels.get("table-tennis", {}).get("resource_dir")
        if not resource_dir:
            return DiagnosticCheck(
                "kernel table-tennis",
                False,
                "No está registrado. Ejecuta scripts/setup_environment.ps1.",
            )
        kernel_json = Path(resource_dir) / "kernel.json"
        argv = json.loads(kernel_json.read_text(encoding="utf-8")).get("argv", [])
        if not argv:
            return DiagnosticCheck("kernel table-tennis", False, "kernel.json no contiene argv.")
        configured = Path(argv[0]).resolve()
        expected = Path(sys.executable).resolve()
        return DiagnosticCheck(
            "kernel table-tennis",
            configured == expected,
            f"{configured} (esperado: {expected})",
        )
    except Exception as exc:
        return DiagnosticCheck("kernel table-tennis", False, str(exc))


def _output_check() -> DiagnosticCheck:
    path = Path("outputs")
    probe = path / ".doctor-write-test"
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return DiagnosticCheck("outputs escribible", True, str(path.resolve()))
    except Exception as exc:
        return DiagnosticCheck("outputs escribible", False, str(exc))


def collect_diagnostics() -> list[DiagnosticCheck]:
    """Collect all environment checks without raising on individual failures."""

    try:
        import table_tennis

        package = DiagnosticCheck(
            "table_tennis",
            True,
            str(Path(table_tennis.__file__).resolve()),
        )
    except Exception as exc:
        package = DiagnosticCheck("table_tennis", False, str(exc))

    checks = [
        DiagnosticCheck("Python", sys.version_info >= (3, 10), sys.executable),
        package,
        _dependency_check("numpy", "numpy"),
        _dependency_check("scipy", "scipy"),
        _dependency_check("matplotlib", "matplotlib"),
        _dependency_check("ipywidgets", "ipywidgets"),
        _dependency_check("jupyterlab", "jupyterlab"),
        DiagnosticCheck("FFmpeg", shutil.which("ffmpeg") is not None, shutil.which("ffmpeg") or "no encontrado"),
        DiagnosticCheck("FFprobe", shutil.which("ffprobe") is not None, shutil.which("ffprobe") or "no encontrado"),
        _kernel_check(),
        _output_check(),
    ]
    return checks


def run_doctor(write: Callable[[str], None] = print) -> bool:
    """Print diagnostics and return whether every check passed."""

    checks = collect_diagnostics()
    for check in checks:
        marker = "OK" if check.passed else "FAIL"
        write(f"[{marker}] {check.name}: {check.detail}")
    passed = all(check.passed for check in checks)
    write(f"summary passed={sum(check.passed for check in checks)}/{len(checks)}")
    return passed
