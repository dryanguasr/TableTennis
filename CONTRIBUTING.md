# Guía de evolución

## Límites de arquitectura

- `physics.py` implementa únicamente dinámica e impactos. No debe importar
  Matplotlib, SciPy, widgets, notebooks ni la CLI.
- `models.py` y `constants.py` no dependen de capas superiores.
- `events.py` interpreta resultados, `validation.py` aplica reglas y
  `exchange.py` coordina el intercambio.
- `rally.py` encadena contactos bidireccionales sin introducir dependencias de
  visualización o CLI.
- `search/` puede usar SciPy, pero sus funciones objetivo deben ser
  serializables para funcionar con múltiples procesos en Windows.
- `visualization/`, `benchmarks/` y la CLI consumen la API; el núcleo nunca
  debe importarlos.

## Nuevos servicios y devoluciones

1. Añada el dato reproducible al módulo correspondiente de `presets/`.
2. Exponga un constructor que devuelva modelos tipados, no diccionarios
   dependientes de una interfaz.
3. Añada un caso al benchmark y una prueba de legalidad.
4. Informe error de bote, error de efecto y margen sobre la red.
5. Verifique derecha y revés cuando el gesto de raqueta sea parte del cambio.

No cambie simultáneamente ecuaciones físicas y presets calibrados. Primero
modifique el modelo y añada pruebas físicas; después recalibre en un cambio
separado y documente los errores anteriores y nuevos.

## Nuevos ejercicios

1. Defina el patrón con `ExerciseStroke` en `presets/exercises.py`.
2. Exprese alas y destinos desde la perspectiva del jugador; no codifique
   coordenadas globales dentro del patrón.
3. Mantenga un mínimo de tres vueltas y compruebe la continuidad de posición,
   velocidad y efecto entre todos los contactos.
4. Ejecute `table-tennis search exercise --exercise <nombre>` y revise el JSON
   antes de promover cambios a los presets.
5. Exija bote en el lado objetivo, cruce legal, cero contactos con la red,
   holgura mínima de 5 mm y gesto compatible con el tipo de golpe.
6. Pruebe el lote con `--dry-run`, un MP4 de humo y una segunda ejecución que
   produzca `SKIP`.

La animación de ejercicios debe conservar continuidad de posición y
orientación en los nodos stand by, preparación, impacto y terminación. No
introduzca cambios instantáneos de pose entre la ventana del golpe y el estado
de espera.

## Notebooks y CLI

- Los notebooks importan el paquete instalado; no usan `sys.path` ni copian
  lógica del motor.
- Todos declaran el kernel `table-tennis` y deben mostrar un error accionable
  cuando se abren con otro intérprete.
- El video se guarda bajo `outputs/notebooks/<notebook>/` y se presenta en el
  mismo panel que lo genera.
- La CLI solo traduce argumentos a llamadas de la API.
- Las optimizaciones costosas requieren una acción explícita del usuario.
- Las opciones nuevas deben funcionar también mediante
  `python -m table_tennis`.

## Artefactos

Videos, HTML, JSON de resultados, cachés y archivos temporales se guardan bajo
`outputs/` y no se versionan. Los recursos que forman parte de la
documentación viven en `docs/assets/`.

## Lista mínima antes de integrar

```powershell
python -m unittest discover -s tests -v
python scripts/validate_notebooks.py
table-tennis benchmark returns
table-tennis benchmark direct --repeat 1
table-tennis benchmark racket --repeat 1 --no-video
table-tennis generate benchmark-videos --suite all --dry-run
table-tennis generate exercise-videos --dry-run
git diff --check
```

Para cambios en búsquedas, pruebe `workers=1` y `workers=4`. Para cambios en
notebooks, ejecute `scripts/validate_notebooks.py`; para cambios de video,
compruebe reanudación, FFprobe y al menos un lote con `--workers 2`. No guarde
outputs en los archivos versionados.
