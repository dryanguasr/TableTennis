"""Execute every active notebook and exercise its MP4 presentation action."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
OUTPUT_DIR = ROOT / "outputs" / "notebook-validation"

SMOKE_CELLS = {
    "01_direct_trajectory_explorer.ipynb": """
_started = time.time()
video_duration.value = 1
video_fps.value = 15
on_save(None)
_files = list((Path('outputs/notebooks/01_direct_trajectory_explorer')).glob('*.mp4'))
assert _files and max(path.stat().st_mtime for path in _files) >= _started - 1
_video = max(_files, key=lambda path: path.stat().st_mtime)
assert 'data:video/mp4;base64' in Video(filename=str(_video), embed=True)._repr_html_()
print(f'VIDEO_OK {_video}')
""",
    "02_racket_impact_explorer.ipynb": """
_started = time.time()
video_duration.value = 1
video_fps.value = 15
on_animate_clicked(None)
_files = list((Path('outputs/notebooks/02_racket_impact_explorer')).glob('*.mp4'))
assert _files and max(path.stat().st_mtime for path in _files) >= _started - 1
_video = max(_files, key=lambda path: path.stat().st_mtime)
assert 'data:video/mp4;base64' in Video(filename=str(_video), embed=True)._repr_html_()
print(f'VIDEO_OK {_video}')
""",
    "03_service_parameter_search.ipynb": """
_started = time.time()
service.value = 'reverse_pendulum'
depth.value = 'short'
lane.value = 'forehand'
apply_preset()
maxiter.value = 1
popsize.value = 3
polish.value = False
workers.value = 1
t_max.value = 2
on_run(None)
assert last_search and last_search['result'].success
video_fps.value = 15
on_video(None)
_files = list((Path('outputs/notebooks/03_service_parameter_search')).glob('*.mp4'))
assert _files and max(path.stat().st_mtime for path in _files) >= _started - 1
_video = max(_files, key=lambda path: path.stat().st_mtime)
assert 'data:video/mp4;base64' in Video(filename=str(_video), embed=True)._repr_html_()
print(f'VIDEO_OK {_video}')
""",
    "04_serve_return_search.ipynb": """
_started = time.time()
maxiter.value = 1
popsize.value = 4
restarts.value = 1
workers.value = 1
run_search(None)
assert last_exchange and last_exchange['return_search'].validation.passed
video_duration.value = 2
video_fps.value = 15
generate_video(None)
_files = list((Path('outputs/notebooks/04_serve_return_search')).glob('*.mp4'))
assert _files and max(path.stat().st_mtime for path in _files) >= _started - 1
_video = max(_files, key=lambda path: path.stat().st_mtime)
assert 'data:video/mp4;base64' in Video(filename=str(_video), embed=True)._repr_html_()
print(f'VIDEO_OK {_video}')
""",
}


def validate_notebook(path: Path, timeout: int) -> Path:
    notebook = nbformat.read(path, as_version=4)
    smoke = nbformat.v4.new_code_cell(
        "import time\n" + SMOKE_CELLS[path.name].strip() + "\n"
    )
    notebook.cells.append(smoke)
    client = NotebookClient(
        notebook,
        kernel_name="table-tennis",
        timeout=timeout,
        allow_errors=False,
    )
    client.execute(cwd=str(ROOT))
    output = OUTPUT_DIR / path.name
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", action="append", choices=sorted(SMOKE_CELLS))
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("IPYTHONDIR", str(ROOT / "outputs" / "ipython"))
    selected = args.notebook or sorted(SMOKE_CELLS)
    for index, name in enumerate(selected, start=1):
        print(f"[{index}/{len(selected)}] EXECUTE {name}", flush=True)
        output = validate_notebook(NOTEBOOK_DIR / name, args.timeout)
        print(f"[{index}/{len(selected)}] OK {output}", flush=True)


if __name__ == "__main__":
    main()
