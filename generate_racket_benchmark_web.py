"""Generate a static viewer for racket-impact benchmark serve videos."""

from __future__ import annotations

import json
from pathlib import Path

from benchmark_racket_services import build_cases, case_filename


SERVICE_LABELS = {
    "pendulum": "Pendulum",
    "reverse_pendulum": "Reverse pendulum",
    "hook": "Hook",
    "tomahawk": "Tomahawk",
    "reverse_tomahawk": "Reverse tomahawk",
    "backhand_standard": "Backhand standard",
}

DEPTH_LABELS = {
    "short": "Short",
    "two_bounce": "Two bounce",
    "long": "Long",
}

LANE_LABELS = {
    "forehand": "Forehand",
    "elbow": "Elbow",
    "backhand": "Backhand",
}


def rounded_tuple(values, digits: int = 2) -> list[float]:
    return [round(float(value), digits) for value in values]


def build_viewer_data(video_dir: Path) -> list[dict[str, object]]:
    cases = []
    for case in build_cases():
        params = case.params
        filename = case_filename(case)
        video_path = video_dir / filename
        cases.append(
            {
                "id": f"{case.service}_{case.depth}_{case.lane}",
                "service": case.service,
                "depth": case.depth,
                "lane": case.lane,
                "serviceLabel": SERVICE_LABELS.get(case.service, case.service),
                "depthLabel": DEPTH_LABELS.get(case.depth, case.depth),
                "laneLabel": LANE_LABELS.get(case.lane, case.lane),
                "title": (
                    f"{SERVICE_LABELS.get(case.service, case.service)} / "
                    f"{DEPTH_LABELS.get(case.depth, case.depth)} / "
                    f"{LANE_LABELS.get(case.lane, case.lane)}"
                ),
                "video": str(video_path.as_posix()),
                "videoExists": video_path.exists(),
                "ballPosition": rounded_tuple(params.ball_position),
                "incomingBallVelocity": rounded_tuple(params.ball_velocity),
                "incomingBallOmega": rounded_tuple(params.ball_omega),
                "rubberFriction": round(float(params.rubber_friction), 4),
                "rubberRestitution": round(float(params.rubber_restitution), 4),
                "racketAngle": rounded_tuple(params.racket_angle),
                "racketVelocity": rounded_tuple(params.racket_velocity),
            }
        )
    return cases


HTML_TEMPLATE = """<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Benchmark de Servicios con Raqueta</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #10130f;
      --panel: rgba(255, 255, 255, .94);
      --panel-dark: rgba(16, 19, 15, .82);
      --ink: #172018;
      --ink-light: #f3f7ef;
      --muted: #6c776b;
      --muted-light: #b9c5b5;
      --line: rgba(255, 255, 255, .18);
      --line-dark: rgba(18, 31, 19, .14);
      --accent: #17a86b;
      --accent-strong: #09784c;
      --racket: #e94c3d;
      --table: #1f7a55;
      --gold: #f4c95d;
      --good: #1f8a58;
      --warn: #b55f00;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      color: var(--ink-light);
      background:
        linear-gradient(115deg, rgba(31, 122, 85, .22) 0 1px, transparent 1px 120px),
        linear-gradient(155deg, rgba(233, 76, 61, .14) 0 1px, transparent 1px 150px),
        radial-gradient(circle at 78% 10%, rgba(244, 201, 93, .16), transparent 28%),
        linear-gradient(135deg, #10130f 0%, #142317 48%, #191815 100%);
      background-attachment: fixed;
    }}

    .app {{
      min-height: 100vh;
      display: grid;
      grid-template-columns: minmax(320px, 410px) 1fr;
    }}

    aside {{
      border-right: 1px solid var(--line);
      background:
        linear-gradient(180deg, rgba(255, 255, 255, .16), rgba(255, 255, 255, .08)),
        rgba(12, 15, 12, .78);
      backdrop-filter: blur(18px);
      padding: 20px;
      overflow: auto;
      max-height: 100vh;
      box-shadow: 20px 0 60px rgba(0, 0, 0, .24);
    }}

    main {{
      padding: 26px;
      min-width: 0;
    }}

    h1 {{
      color: #fff;
      font-size: 28px;
      line-height: 1.2;
      margin: 0 0 6px;
      text-wrap: balance;
    }}

    .subtitle {{
      color: var(--muted-light);
      font-size: 13px;
      margin: 0 0 20px;
    }}

    label {{
      display: block;
      color: #dce8d8;
      font-size: 12px;
      font-weight: 700;
      margin-bottom: 6px;
      text-transform: uppercase;
      letter-spacing: .04em;
    }}

    select, input {{
      width: 100%;
      border: 1px solid rgba(255, 255, 255, .18);
      border-radius: 8px;
      background: rgba(255, 255, 255, .95);
      color: var(--ink);
      font: inherit;
      padding: 10px 11px;
      box-shadow: 0 10px 30px rgba(0, 0, 0, .12);
    }}

    .filters {{
      display: grid;
      gap: 12px;
      margin-bottom: 16px;
    }}

    .toolbar {{
      display: flex;
      gap: 8px;
      margin-bottom: 14px;
    }}

    button {{
      border: 1px solid rgba(255, 255, 255, .18);
      border-radius: 8px;
      background: linear-gradient(135deg, rgba(255, 255, 255, .98), rgba(235, 250, 240, .98));
      color: var(--accent-strong);
      font: inherit;
      font-weight: 800;
      padding: 9px 12px;
      cursor: pointer;
      box-shadow: 0 10px 30px rgba(0, 0, 0, .14);
    }}

    button:hover, button.active {{
      border-color: rgba(244, 201, 93, .75);
      color: #0b5f3e;
      transform: translateY(-1px);
    }}

    .case-list {{
      display: grid;
      gap: 8px;
    }}

    .case-card {{
      border: 1px solid rgba(255, 255, 255, .16);
      border-radius: 8px;
      background: rgba(255, 255, 255, .09);
      color: var(--ink-light);
      padding: 11px;
      cursor: pointer;
      transition: border-color .18s ease, background .18s ease, transform .18s ease;
    }}

    .case-card.active {{
      border-color: rgba(244, 201, 93, .9);
      background: linear-gradient(135deg, rgba(31, 122, 85, .52), rgba(233, 76, 61, .24));
      box-shadow: 0 0 0 2px rgba(244, 201, 93, .18), 0 18px 38px rgba(0, 0, 0, .22);
      transform: translateX(2px);
    }}

    .case-card:hover {{
      background: rgba(255, 255, 255, .14);
    }}

    .case-card-title {{
      font-size: 14px;
      font-weight: 750;
      margin-bottom: 6px;
    }}

    .case-card-meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }}

    .pill {{
      border-radius: 999px;
      background: rgba(255, 255, 255, .86);
      color: #254234;
      font-size: 12px;
      font-weight: 760;
      padding: 4px 9px;
    }}

    .pill.good {{
      background: #e6f4ed;
      color: var(--good);
    }}

    .pill.warn {{
      background: #fff2dc;
      color: var(--warn);
    }}

    .viewer {{
      display: grid;
      gap: 18px;
    }}

    .video-shell {{
      position: relative;
      background:
        linear-gradient(135deg, rgba(31, 122, 85, .28), rgba(233, 76, 61, .18)),
        #0b0f0d;
      border-radius: 18px;
      overflow: hidden;
      aspect-ratio: 16 / 10;
      display: grid;
      place-items: center;
      border: 1px solid rgba(255, 255, 255, .2);
      box-shadow: 0 28px 80px rgba(0, 0, 0, .34), inset 0 0 0 1px rgba(255, 255, 255, .08);
    }}

    .video-shell::after {{
      content: "";
      position: absolute;
      inset: 0;
      pointer-events: none;
      box-shadow: inset 0 0 0 1px rgba(255, 255, 255, .12), inset 0 -80px 100px rgba(0, 0, 0, .2);
    }}

    video {{
      width: 100%;
      height: 100%;
      object-fit: cover;
      object-position: 50% 100%;
      background: #0b0f0d;
    }}

    .missing {{
      color: #fff;
      text-align: center;
      padding: 24px;
    }}

    .header-row {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 16px;
      padding: 18px;
      border: 1px solid rgba(255, 255, 255, .16);
      border-radius: 16px;
      background: linear-gradient(135deg, rgba(255, 255, 255, .18), rgba(255, 255, 255, .08));
      backdrop-filter: blur(16px);
      box-shadow: 0 24px 70px rgba(0, 0, 0, .24);
    }}

    .title-block h2 {{
      margin: 0;
      color: #fff;
      font-size: clamp(26px, 3.3vw, 44px);
      letter-spacing: 0;
      text-wrap: balance;
    }}

    .title-block p {{
      margin: 6px 0 0;
      color: var(--muted-light);
    }}

    .stats {{
      display: grid;
      grid-template-columns: repeat(4, minmax(130px, 1fr));
      gap: 12px;
    }}

    .stat, .params {{
      border: 1px solid rgba(255, 255, 255, .16);
      border-radius: 14px;
      background: var(--panel);
      color: var(--ink);
      padding: 14px;
      box-shadow: 0 18px 50px rgba(0, 0, 0, .18);
    }}

    .stat span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 4px;
    }}

    .stat strong {{
      font-size: 20px;
      color: #0e3826;
    }}

    .params {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px 18px;
    }}

    .param-group h3 {{
      font-size: 15px;
      margin: 0 0 10px;
      color: #0d3e2a;
    }}

    dl {{
      display: grid;
      grid-template-columns: 140px 1fr;
      gap: 6px 10px;
      margin: 0;
      font-size: 13px;
    }}

    dt {{
      color: var(--muted);
    }}

    dd {{
      margin: 0;
      font-family: ui-monospace, "Cascadia Code", Consolas, monospace;
      overflow-wrap: anywhere;
    }}

    @media (max-width: 900px) {{
      .app {{
        grid-template-columns: 1fr;
      }}

      aside {{
        max-height: none;
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }}

      .stats, .params {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <h1>Benchmark con raqueta</h1>
      <p class="subtitle">54 servicios animados con los parámetros de `benchmark_racket_services.py`.</p>
      <div class="filters">
        <div>
          <label for="serviceFilter">Servicio</label>
          <select id="serviceFilter"></select>
        </div>
        <div>
          <label for="depthFilter">Profundidad</label>
          <select id="depthFilter"></select>
        </div>
        <div>
          <label for="laneFilter">Zona</label>
          <select id="laneFilter"></select>
        </div>
        <div>
          <label for="searchInput">Buscar</label>
          <input id="searchInput" type="search" placeholder="pendulum, backhand, long...">
        </div>
      </div>
      <div class="toolbar">
        <button id="playAllButton" type="button">Reproducir</button>
        <button id="resetButton" type="button">Limpiar</button>
      </div>
      <div id="caseList" class="case-list"></div>
    </aside>

    <main>
      <section class="viewer">
        <div class="header-row">
          <div class="title-block">
            <h2 id="caseTitle"></h2>
            <p id="caseSubtitle"></p>
          </div>
          <span id="videoStatus" class="pill"></span>
        </div>
        <div id="videoShell" class="video-shell"></div>
        <div class="stats">
          <div class="stat"><span>Fricción goma</span><strong id="frictionStat"></strong></div>
          <div class="stat"><span>Restitución goma</span><strong id="restitutionStat"></strong></div>
          <div class="stat"><span>Velocidad entrante Z</span><strong id="incomingZStat"></strong></div>
          <div class="stat"><span>Velocidad raqueta X</span><strong id="racketXStat"></strong></div>
        </div>
        <div class="params">
          <div class="param-group">
            <h3>Bola antes del impacto</h3>
            <dl>
              <dt>Posición</dt><dd id="ballPosition"></dd>
              <dt>Velocidad</dt><dd id="incomingVelocity"></dd>
              <dt>Spin</dt><dd id="incomingSpin"></dd>
            </dl>
          </div>
          <div class="param-group">
            <h3>Raqueta y goma</h3>
            <dl>
              <dt>Ángulo</dt><dd id="racketAngle"></dd>
              <dt>Velocidad</dt><dd id="racketVelocity"></dd>
              <dt>Goma</dt><dd id="rubber"></dd>
            </dl>
          </div>
        </div>
      </section>
    </main>
  </div>

  <script>
    const cases = {cases_json};
    const labels = {{
      service: {service_labels_json},
      depth: {depth_labels_json},
      lane: {lane_labels_json}
    }};
    const state = {{
      service: "all",
      depth: "all",
      lane: "all",
      search: "",
      selectedId: cases[0]?.id
    }};

    const $ = (id) => document.getElementById(id);
    const formatVector = (values, unit = "mm/s") => values.map((value) => Number(value).toLocaleString("es-CO", {{ maximumFractionDigits: 2 }})).join(", ") + ` ${{unit}}`;

    function fillSelect(id, field, labelMap) {{
      const select = $(id);
      const values = [...new Set(cases.map((item) => item[field]))];
      select.innerHTML = `<option value="all">Todos</option>` + values.map((value) => `<option value="${{value}}">${{labelMap[value] ?? value}}</option>`).join("");
      select.addEventListener("change", () => {{
        state[field] = select.value;
        renderList();
      }});
    }}

    function filteredCases() {{
      const query = state.search.trim().toLowerCase();
      return cases.filter((item) => {{
        const matchesFilters =
          (state.service === "all" || item.service === state.service) &&
          (state.depth === "all" || item.depth === state.depth) &&
          (state.lane === "all" || item.lane === state.lane);
        const haystack = `${{item.title}} ${{item.id}}`.toLowerCase();
        return matchesFilters && (!query || haystack.includes(query));
      }});
    }}

    function selectCase(id) {{
      state.selectedId = id;
      renderSelected();
      renderList();
    }}

    function renderList() {{
      const list = $("caseList");
      const items = filteredCases();
      if (!items.length) {{
        list.innerHTML = `<div class="case-card">No hay servicios con esos filtros.</div>`;
        return;
      }}
      if (!items.some((item) => item.id === state.selectedId)) {{
        state.selectedId = items[0].id;
        renderSelected();
      }}
      list.innerHTML = items.map((item) => `
        <article class="case-card ${{item.id === state.selectedId ? "active" : ""}}" data-id="${{item.id}}">
          <div class="case-card-title">${{item.serviceLabel}}</div>
          <div class="case-card-meta">
            <span class="pill">${{item.depthLabel}}</span>
            <span class="pill">${{item.laneLabel}}</span>
            <span class="pill ${{item.videoExists ? "good" : "warn"}}">${{item.videoExists ? "MP4" : "Sin MP4"}}</span>
          </div>
        </article>
      `).join("");
      list.querySelectorAll(".case-card[data-id]").forEach((card) => {{
        card.addEventListener("click", () => selectCase(card.dataset.id));
      }});
    }}

    function renderSelected() {{
      const item = cases.find((candidate) => candidate.id === state.selectedId) ?? cases[0];
      if (!item) return;

      $("caseTitle").textContent = item.title;
      $("caseSubtitle").textContent = `${{item.id}}.mp4`;
      $("videoStatus").textContent = item.videoExists ? "Video disponible" : "Video no encontrado";
      $("videoStatus").className = `pill ${{item.videoExists ? "good" : "warn"}}`;

      $("videoShell").innerHTML = item.videoExists
        ? `<video id="serviceVideo" src="${{item.video}}" controls autoplay muted loop playsinline></video>`
        : `<div class="missing">No encontré el archivo <code>${{item.video}}</code>. Genera los videos con <code>python benchmark_racket_services.py</code>.</div>`;

      $("frictionStat").textContent = item.rubberFriction.toFixed(2);
      $("restitutionStat").textContent = item.rubberRestitution.toFixed(2);
      $("incomingZStat").textContent = `${{Math.round(item.incomingBallVelocity[2]).toLocaleString("es-CO")}} mm/s`;
      $("racketXStat").textContent = `${{Math.round(item.racketVelocity[0]).toLocaleString("es-CO")}} mm/s`;

      $("ballPosition").textContent = formatVector(item.ballPosition, "mm");
      $("incomingVelocity").textContent = formatVector(item.incomingBallVelocity);
      $("incomingSpin").textContent = formatVector(item.incomingBallOmega, "rad/s");
      $("racketAngle").textContent = formatVector(item.racketAngle, "deg");
      $("racketVelocity").textContent = formatVector(item.racketVelocity);
      $("rubber").textContent = `fricción=${{item.rubberFriction}}, restitución=${{item.rubberRestitution}}`;
    }}

    fillSelect("serviceFilter", "service", labels.service);
    fillSelect("depthFilter", "depth", labels.depth);
    fillSelect("laneFilter", "lane", labels.lane);

    $("searchInput").addEventListener("input", (event) => {{
      state.search = event.target.value;
      renderList();
    }});

    $("resetButton").addEventListener("click", () => {{
      state.service = "all";
      state.depth = "all";
      state.lane = "all";
      state.search = "";
      $("serviceFilter").value = "all";
      $("depthFilter").value = "all";
      $("laneFilter").value = "all";
      $("searchInput").value = "";
      renderList();
    }});

    $("playAllButton").addEventListener("click", () => {{
      const video = $("serviceVideo");
      if (video) video.play();
    }});

    renderSelected();
    renderList();
  </script>
</body>
</html>
"""


def main() -> None:
    output = Path("racket_benchmark_viewer.html")
    video_dir = Path("benchmark_racket_services")
    cases = build_viewer_data(video_dir)
    html = HTML_TEMPLATE.format(
        cases_json=json.dumps(cases, ensure_ascii=False, indent=4),
        service_labels_json=json.dumps(SERVICE_LABELS, ensure_ascii=False, indent=4),
        depth_labels_json=json.dumps(DEPTH_LABELS, ensure_ascii=False, indent=4),
        lane_labels_json=json.dumps(LANE_LABELS, ensure_ascii=False, indent=4),
    )
    output.write_text(html, encoding="utf-8")
    available = sum(1 for case in cases if case["videoExists"])
    print(f"Wrote {output} with {len(cases)} cases and {available} available videos.")


if __name__ == "__main__":
    main()
