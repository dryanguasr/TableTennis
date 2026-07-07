"""Generate a static web viewer for the multi-stroke exercise catalog."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ..presets.exercises import build_exercises


FAMILY_BY_EXERCISE = {
    "drive_to_drive": "continuity",
    "backhand_to_backhand": "continuity",
    "figure_eight": "continuity",
    "falkenberg": "footwork",
    "hit_and_pass_forehand": "rhythm",
    "hit_and_pass_backhand": "rhythm",
    "third_ball_hit_and_pass_forehand": "third_ball",
    "third_ball_hit_and_pass_backhand": "third_ball",
    "third_ball_change_of_pace": "third_ball",
    "third_ball_change_of_pace_backhand": "third_ball",
}

FAMILY_LABELS = {
    "continuity": "Continuidad",
    "footwork": "Desplazamiento",
    "rhythm": "Ritmo",
    "third_ball": "Tercera bola",
}

WING_BY_EXERCISE = {
    "drive_to_drive": "forehand",
    "backhand_to_backhand": "backhand",
    "figure_eight": "mixed",
    "falkenberg": "mixed",
    "hit_and_pass_forehand": "forehand",
    "hit_and_pass_backhand": "backhand",
    "third_ball_hit_and_pass_forehand": "forehand",
    "third_ball_hit_and_pass_backhand": "backhand",
    "third_ball_change_of_pace": "forehand",
    "third_ball_change_of_pace_backhand": "backhand",
}

WING_LABELS = {
    "forehand": "Drive",
    "backhand": "Revés",
    "mixed": "Mixto",
    "elbow": "Codo",
}

KIND_LABELS = {
    "drive": "Ataque",
    "topspin": "Topspin",
    "block": "Bloqueo",
    "push": "Corte",
    "chop": "Corte defensivo",
    "serve": "Servicio",
}

DEPTH_LABELS = {
    "short": "Corto",
    "long": "Largo",
}


def build_exercise_viewer_data(
    video_dir: Path,
    viewer_dir: Path = Path("."),
) -> list[dict[str, object]]:
    """Describe all exercises without running simulations or embedding videos."""

    items: list[dict[str, object]] = []
    for definition in build_exercises(cycles=3):
        video_path = video_dir / f"{definition.name}.mp4"
        sequence = []
        for index, stroke in enumerate(definition.strokes, start=1):
            sequence.append(
                {
                    "index": index,
                    "label": stroke.label or KIND_LABELS[stroke.kind],
                    "kind": stroke.kind,
                    "kindLabel": KIND_LABELS[stroke.kind],
                    "hitter": "Jugador A" if stroke.hitter == "near" else "Jugador B",
                    "wing": stroke.wing,
                    "wingLabel": WING_LABELS.get(stroke.wing, stroke.wing),
                    "targetWing": WING_LABELS.get(
                        stroke.target_wing,
                        stroke.target_wing,
                    ),
                    "moment": stroke.contact_moment,
                    "depth": stroke.depth,
                    "depthLabel": DEPTH_LABELS[stroke.depth],
                    "openingAttack": stroke.opening_attack,
                    "cycle": stroke.cycle,
                }
            )
        family = FAMILY_BY_EXERCISE[definition.name]
        wing = WING_BY_EXERCISE[definition.name]
        items.append(
            {
                "id": definition.name,
                "title": definition.title,
                "family": family,
                "familyLabel": FAMILY_LABELS[family],
                "wing": wing,
                "wingLabel": WING_LABELS[wing],
                "cycles": definition.cycles,
                "strokeCount": len(definition.strokes),
                "video": Path(
                    os.path.relpath(video_path, start=viewer_dir)
                ).as_posix(),
                "videoExists": video_path.is_file(),
                "generateCommand": (
                    "table-tennis generate exercise-videos "
                    f"--exercise {definition.name} --overwrite"
                ),
                "sequence": sequence,
            }
        )
    return items


HTML_TEMPLATE = """<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Ejercicios de tenis de mesa</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0b1110;
      --panel: #131c19;
      --panel-2: #192521;
      --line: #2c3d37;
      --text: #eef6f1;
      --muted: #a4b4ad;
      --accent: #51d393;
      --accent-2: #e9bf61;
      --danger: #ff8c7e;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background:
        radial-gradient(circle at 80% 5%, #18392c 0, transparent 35rem),
        var(--bg);
      color: var(--text);
      font: 15px/1.45 Inter, ui-sans-serif, system-ui, sans-serif;
    }
    .app { display: grid; grid-template-columns: 340px 1fr; min-height: 100vh; }
    aside {
      border-right: 1px solid var(--line);
      background: rgba(11, 17, 16, .88);
      padding: 24px;
      overflow: auto;
      max-height: 100vh;
      position: sticky;
      top: 0;
    }
    h1 { margin: 0 0 5px; font-size: 22px; }
    .intro, .muted { color: var(--muted); }
    .filters { display: grid; gap: 9px; margin: 22px 0 18px; }
    input, select, button {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: var(--panel);
      color: var(--text);
      padding: 10px 12px;
      font: inherit;
    }
    button { cursor: pointer; }
    button:hover { border-color: var(--accent); }
    .exercise-list { display: grid; gap: 8px; }
    .exercise-card {
      text-align: left;
      background: transparent;
      padding: 12px;
    }
    .exercise-card.active {
      background: var(--panel-2);
      border-color: var(--accent);
    }
    .card-title { display: block; font-weight: 700; }
    .card-meta { display: flex; justify-content: space-between; margin-top: 5px; }
    .status.ok { color: var(--accent); }
    .status.missing { color: var(--danger); }
    main { min-width: 0; padding: clamp(22px, 4vw, 56px); }
    .hero { display: flex; align-items: end; justify-content: space-between; gap: 18px; }
    .hero h2 { margin: 5px 0 0; font-size: clamp(28px, 4vw, 48px); }
    .eyebrow { color: var(--accent); font-weight: 800; letter-spacing: .08em; text-transform: uppercase; }
    .pills { display: flex; flex-wrap: wrap; gap: 8px; margin: 18px 0; }
    .pill { padding: 6px 10px; border: 1px solid var(--line); border-radius: 999px; color: var(--muted); }
    .video-shell {
      aspect-ratio: 16 / 9;
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 18px;
      overflow: hidden;
      background: #050807;
      display: grid;
      place-items: center;
    }
    video { width: 100%; height: 100%; object-fit: contain; }
    .missing { max-width: 680px; padding: 30px; text-align: center; color: var(--muted); }
    code { color: var(--accent-2); overflow-wrap: anywhere; }
    .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 18px 0 30px; }
    .stat { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 14px; }
    .stat strong { display: block; font-size: 22px; }
    .sequence { display: grid; gap: 8px; }
    .stroke {
      display: grid;
      grid-template-columns: 42px minmax(150px, 1.5fr) repeat(4, minmax(90px, 1fr));
      gap: 10px;
      align-items: center;
      border: 1px solid var(--line);
      border-radius: 11px;
      background: var(--panel);
      padding: 10px 12px;
    }
    .stroke .number { color: var(--accent); font-weight: 800; }
    .stroke small { color: var(--muted); }
    @media (max-width: 900px) {
      .app { grid-template-columns: 1fr; }
      aside { max-height: none; position: static; border-right: 0; border-bottom: 1px solid var(--line); }
      .exercise-list { grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); }
      .stroke { grid-template-columns: 36px 1fr 1fr; }
      .stroke span:nth-child(n+4) { display: none; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <h1>Ejercicios</h1>
      <div class="intro">Catálogo físico de rallies multigolpe.</div>
      <div class="filters">
        <input id="search" type="search" placeholder="Buscar ejercicio…">
        <select id="family">
          <option value="all">Todas las familias</option>
          <option value="continuity">Continuidad</option>
          <option value="footwork">Desplazamiento</option>
          <option value="rhythm">Ritmo</option>
          <option value="third_ball">Tercera bola</option>
        </select>
        <select id="wing">
          <option value="all">Todas las alas</option>
          <option value="forehand">Drive</option>
          <option value="backhand">Revés</option>
          <option value="mixed">Mixto</option>
        </select>
        <button id="reset" type="button">Limpiar filtros</button>
      </div>
      <div id="list" class="exercise-list"></div>
    </aside>
    <main>
      <header class="hero">
        <div>
          <div id="familyLabel" class="eyebrow"></div>
          <h2 id="title"></h2>
        </div>
      </header>
      <div id="pills" class="pills"></div>
      <div id="videoShell" class="video-shell"></div>
      <div class="stats">
        <div class="stat"><strong id="cycles"></strong><span class="muted">vueltas</span></div>
        <div class="stat"><strong id="strokes"></strong><span class="muted">golpes</span></div>
        <div class="stat"><strong id="videoState"></strong><span class="muted">estado del MP4</span></div>
      </div>
      <h3>Secuencia ordenada</h3>
      <div id="sequence" class="sequence"></div>
    </main>
  </div>
  <script>
    const exercises = __EXERCISES_JSON__;
    const state = { selected: exercises[0]?.id, search: "", family: "all", wing: "all" };
    const $ = id => document.getElementById(id);
    const escapeHtml = value => String(value).replace(/[&<>"']/g, char => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
    })[char]);

    function filteredExercises() {
      const needle = state.search.trim().toLocaleLowerCase("es");
      return exercises.filter(item =>
        (state.family === "all" || item.family === state.family) &&
        (state.wing === "all" || item.wing === state.wing) &&
        (!needle || `${item.title} ${item.familyLabel} ${item.wingLabel}`.toLocaleLowerCase("es").includes(needle))
      );
    }

    function renderList() {
      const visible = filteredExercises();
      if (!visible.some(item => item.id === state.selected)) state.selected = visible[0]?.id;
      $("list").innerHTML = visible.length
        ? visible.map(item => `
          <button class="exercise-card ${item.id === state.selected ? "active" : ""}" data-id="${item.id}">
            <span class="card-title">${escapeHtml(item.title)}</span>
            <span class="card-meta">
              <small>${escapeHtml(item.familyLabel)} · ${escapeHtml(item.wingLabel)}</small>
              <small class="status ${item.videoExists ? "ok" : "missing"}">${item.videoExists ? "MP4 listo" : "Falta MP4"}</small>
            </span>
          </button>`).join("")
        : `<div class="muted">No hay ejercicios con esos filtros.</div>`;
      document.querySelectorAll(".exercise-card").forEach(button => {
        button.addEventListener("click", () => {
          state.selected = button.dataset.id;
          renderList();
          renderSelected();
        });
      });
      renderSelected();
    }

    function renderSelected() {
      const item = exercises.find(candidate => candidate.id === state.selected);
      if (!item) {
        $("title").textContent = "Sin resultados";
        $("videoShell").innerHTML = `<div class="missing">Ajusta los filtros para seleccionar un ejercicio.</div>`;
        $("sequence").innerHTML = "";
        return;
      }
      $("familyLabel").textContent = item.familyLabel;
      $("title").textContent = item.title;
      $("pills").innerHTML = `
        <span class="pill">${escapeHtml(item.wingLabel)}</span>
        <span class="pill">${item.cycles} vueltas</span>
        <span class="pill">${item.strokeCount} golpes</span>`;
      $("videoShell").innerHTML = item.videoExists
        ? `<video src="${escapeHtml(item.video)}" controls autoplay muted loop playsinline></video>`
        : `<div class="missing">No se encontró <code>${escapeHtml(item.video)}</code>.<br>Genéralo con:<br><code>${escapeHtml(item.generateCommand)}</code></div>`;
      $("cycles").textContent = item.cycles;
      $("strokes").textContent = item.strokeCount;
      $("videoState").textContent = item.videoExists ? "Disponible" : "Ausente";
      $("sequence").innerHTML = item.sequence.map(stroke => `
        <div class="stroke">
          <span class="number">${stroke.index}</span>
          <span><strong>${escapeHtml(stroke.label)}</strong><br><small>${escapeHtml(stroke.kindLabel)}</small></span>
          <span>${escapeHtml(stroke.hitter)}</span>
          <span>Punto ${stroke.moment}</span>
          <span>${escapeHtml(stroke.depthLabel)}</span>
          <span>A ${escapeHtml(stroke.targetWing)}</span>
        </div>`).join("");
    }

    $("search").addEventListener("input", event => { state.search = event.target.value; renderList(); });
    $("family").addEventListener("change", event => { state.family = event.target.value; renderList(); });
    $("wing").addEventListener("change", event => { state.wing = event.target.value; renderList(); });
    $("reset").addEventListener("click", () => {
      state.search = ""; state.family = "all"; state.wing = "all";
      $("search").value = ""; $("family").value = "all"; $("wing").value = "all";
      renderList();
    });
    renderList();
  </script>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="table-tennis generate exercise-viewer",
        description=__doc__,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/viewers/exercise_viewer.html"),
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=Path("outputs/exercises"),
    )
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    items = build_exercise_viewer_data(args.video_dir, args.output.parent)
    html = HTML_TEMPLATE.replace(
        "__EXERCISES_JSON__",
        json.dumps(items, ensure_ascii=False, indent=2),
    )
    args.output.write_text(html, encoding="utf-8")
    available = sum(bool(item["videoExists"]) for item in items)
    print(
        f"Wrote {args.output} with {len(items)} exercises "
        f"and {available} available videos."
    )


if __name__ == "__main__":
    main()
