# Table Tennis Physics Simulator (Work in Progress)

![Table Tennis Simulation](Backhand_composite_service.gif)

This repository explores table-tennis serve physics in Python and MATLAB. It simulates ball flight with drag, Magnus force, table friction, net interaction, spin decay, and an optional racket-impact model that converts racket motion into post-impact ball conditions.

## Key Features

- **3D ball physics:** Simulate table-tennis trajectories with gravity, drag, Magnus lift/sideswerve, spin decay, table bounces, and net collision.
- **Direct serve benchmarks:** Run 54 tuned serve cases from direct ball initial conditions across service family, depth, and lane.
- **Racket-impact benchmarks:** Run the matching 54 racket-impact cases. These derive racket velocity and pre-impact spin from the direct benchmark targets and render synchronized racket motion.
- **Serve-return exchange:** Chain a legal reverse-pendulum serve with a searched forehand or backhand return at trajectory point 2, 3, or 4.
- **Return benchmark:** Validate ten descending-phase return presets across short, two-bounce, and long backspin/topspin responses.
- **Interactive notebooks:** Explore direct-service parameters, racket-impact parameters, benchmark presets, and embedded animations with widgets.
- **Static video viewer:** Browse the racket benchmark MP4s in a local HTML page with filters and parameter readouts.
- **Parameter search tool:** Use a SciPy-powered optimizer to discover serve parameters that satisfy bounce, lane, height, and depth constraints.

## Files

- `table_tennis_simulation.py`: Core physics, plotting, racket drawing, generic animation, and racket-impact animation.
- `benchmark_direct_services.py`: Competitive direct-service benchmark cases.
- `benchmark_racket_services.py`: Racket-impact benchmark cases derived from the direct benchmark.
- `return_parameter_search.py`: Serve-return API, legality validators, contact selection, fixed-rubber racket-motion search, and service-impact inversion.
- `benchmark_returns.py`: Fast validation and optional retuning of the ten pilot returns.
- `generate_return_videos.py`: Generates full serve-return MP4s in `benchmark_returns/`.
- `interactive_table_tennis.ipynb`: Direct-trajectory sliders plus benchmark preset dropdowns.
- `racket_impact_explorer.ipynb`: Racket-impact sliders, benchmark preset dropdowns, coaching moments, and embedded animation.
- `serve_return_search.ipynb`: Interactive service/return search with effect, depth, direction, contact point, material, tolerance, and search-budget controls.
- `service_parameter_search.py`: Direct and racket parameter optimizer. Uses SciPy when available.
- `service_parameter_search.ipynb`: Widget UI for setting optimization constraints and running the search with live global/polishing progress.
- `generate_racket_benchmark_web.py`: Generates the local benchmark video viewer.
- `racket_benchmark_viewer.html`: Static viewer for racket benchmark videos.

## Requirements

The core simulation needs Python with NumPy and Matplotlib. MP4 export needs FFmpeg. The parameter search script works best with SciPy:

```powershell
pip install numpy matplotlib scipy ipywidgets
```

If SciPy is not installed, `service_parameter_search.py` falls back to a slower random search.

## Quick Start

Run a default simulation:

```powershell
python table_tennis_simulation.py
```

Save a simulation as MP4:

```powershell
python table_tennis_simulation.py --save output.mp4
```

Run direct benchmark cases:

```powershell
python benchmark_direct_services.py
```

Save direct benchmark videos:

```powershell
python benchmark_direct_services.py --video-dir benchmark_videos
```

Run racket-impact benchmark cases and save MP4 videos:

```powershell
python benchmark_racket_services.py
```

Validate the ten serve-return presets:

```powershell
python benchmark_returns.py
```

Generate the ten complete serve-return videos:

```powershell
python generate_return_videos.py
```

Videos are written to the ignored `benchmark_returns/` folder. Render a single smoke-test case or select a profile with:

```powershell
python generate_return_videos.py --limit 1
python generate_return_videos.py --profile cut_short
```

Each MP4 lasts at most 5 seconds and stops earlier only when the ball center falls below floor level (`Z < 0`). Change the cap with `--duration`:

```powershell
python generate_return_videos.py --duration 8 --overwrite
```

If FFmpeg is not on `PATH`, pass it explicitly:

```powershell
python benchmark_racket_services.py --ffmpeg C:\path\to\ffmpeg.exe
```

## Web Viewer

Generate or refresh the racket benchmark viewer:

```powershell
python generate_racket_benchmark_web.py
```

Then open:

```text
racket_benchmark_viewer.html
```

The viewer lists the 54 racket benchmark videos with filters for service type, depth, and lane. It also shows the racket angle, racket velocity, incoming ball velocity, spin, friction, restitution, and contact position used by each case.

## Parameter Search

Search direct initial conditions:

```powershell
python service_parameter_search.py --mode direct --service pendulum --depth short --lane elbow
```

Search racket-impact parameters:

```powershell
python service_parameter_search.py --mode racket --service tomahawk --depth short --lane forehand
```

Useful options:

```powershell
python service_parameter_search.py `
  --mode direct `
  --service reverse_pendulum `
  --depth two_bounce `
  --lane backhand `
  --server-x 520 `
  --opponent-x 1850 `
  --opponent-y 427 `
  --max-height 240 `
  --maxiter 120 `
  --popsize 12 `
  --workers 4 `
  --output search_result.json
```

Open `service_parameter_search.ipynb` for a slider-based interface to the same optimizer. You can set server-side bounce, receiver-side bounce, lane target, maximum height after the net, search budget, and whether the serve should prefer or avoid a second bounce on the receiver side. After pressing **Buscar parámetros**, the notebook disables the button and displays generation-by-generation progress, the best cost, and the local-polishing phase.

## Serve-Return Search

The pilot starts with a legal short reverse-pendulum backspin serve to the elbow. Return targets support:

- numeric post-impact spin on global X/Y/Z axes;
- short, two-bounce, or long depth with an editable landing coordinate;
- forehand, elbow, or backhand direction relative to the receiving player;
- forehand or backhand stroke;
- contact during rising point 2, apex point 3, or descending point 4;
- fixed rubber friction/restitution and adjustable landing/spin tolerances.

Ball-table and ball-racket contacts use impulse-based restitution with a Coulomb friction cap. Tangential impulses can no longer exceed the friction supported by the normal impact. The return validator also rejects:

- cut strokes whose racket moves upward or away from the opponent;
- trajectories with excessive height or net clearance;
- balls that reverse direction after the first bounce;
- short/two-bounce shots whose second bounce moves back toward the net.

The stored cut presets retain some lateral spin from the reverse-pendulum serve and target `(-15, 35, 10)` rps rather than forcing an artificial pure backspin vector.

The original 54 racket-service presets explicitly retain the legacy contact model so they remain numerically identical to their matching direct-service initial conditions. New searches and all serve-return presets use the Coulomb model by default.

Run the notebook for interactive searches:

```text
serve_return_search.ipynb
```

Retune the five return profiles instead of loading the stored presets:

```powershell
python benchmark_returns.py --retune --workers 4
```

## Notebooks

- Open `interactive_table_tennis.ipynb` to experiment with direct position, velocity, and spin. The three dropdowns at the top load benchmark presets.
- Open `racket_impact_explorer.ipynb` to experiment with incoming ball, rubber, racket angle, racket velocity, and benchmark racket presets. It identifies coaching moments 1-6, renders the synchronized pre-impact animation, and includes a block for the ten validated return presets.
- Open `serve_return_search.ipynb` to search and visualize the complete service-reception exchange.

## Contributions and Feedback

Feedback, issues, and pull requests are welcome. This is an exploratory physics project, so empirical tuning notes, references to real service mechanics, and validation cases are especially useful.

## About the Author

Hi, I'm David Yanguas, a passionate table tennis player and physics enthusiast. This project combines my love for both fields and serves as an avenue to share the wonders of physics through the lens of a beloved sport. Connect with me on [LinkedIn](https://www.linkedin.com/in/davidyanguasrojas/) to stay updated on my journey and other projects.

---

**Disclaimer:** This project is intended for educational and entertainment purposes. The simulations are approximations and do not capture every complexity of real table-tennis play.
