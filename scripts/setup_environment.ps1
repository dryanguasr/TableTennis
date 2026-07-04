param(
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"

function Find-BootstrapPython {
    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        return $python.Source
    }

    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        $resolved = & $py.Source -3 -c "import sys; print(sys.executable)"
        if ($LASTEXITCODE -eq 0 -and $resolved) {
            return $resolved.Trim()
        }
    }

    $codexPython = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
    if (Test-Path -LiteralPath $codexPython) {
        return $codexPython
    }

    throw "No se encontro Python. Instala Python 3.10+ o proporciona un entorno .venv existente."
}

if (-not (Test-Path -LiteralPath $VenvPython)) {
    $BootstrapPython = Find-BootstrapPython
    Write-Host "Creando .venv con $BootstrapPython"
    & $BootstrapPython -m venv (Join-Path $Root ".venv")
    if ($LASTEXITCODE -ne 0) {
        throw "No se pudo crear .venv."
    }
}

if (-not $SkipInstall) {
    Write-Host "Instalando TableTennis y dependencias de desarrollo..."
    $EditableTarget = "$Root[dev]"
    & $VenvPython -m pip install --no-build-isolation -e $EditableTarget
    if ($LASTEXITCODE -ne 0) {
        throw "La instalacion editable fallo."
    }
}

Write-Host "Registrando kernel Python (TableTennis)..."
& $VenvPython -m ipykernel install --user --name table-tennis --display-name "Python (TableTennis)"
if ($LASTEXITCODE -ne 0) {
    throw "No se pudo registrar el kernel table-tennis."
}

& $VenvPython -m pip check
if ($LASTEXITCODE -ne 0) {
    throw "pip check encontró dependencias incompatibles."
}

& $VenvPython -m table_tennis doctor
if ($LASTEXITCODE -ne 0) {
    throw "El diagnostico del entorno fallo."
}

Write-Host ""
Write-Host "Entorno listo. Selecciona el kernel 'Python (TableTennis)' en Jupyter o VS Code."
