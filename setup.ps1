# Sonara one-time setup. Run from E:\PA:  ./setup.ps1
$ErrorActionPreference = 'Stop'

Write-Host "== Sonara setup ==" -ForegroundColor Cyan

# 1. Python env (uv pins 3.12 per pyproject; downloads it if missing)
uv sync

# 2. Piper TTS voice (~60MB), skipped if already present.
# Piper won the 2026-07-30 bake-off on this CPU (~1s/sentence vs Kokoro's 2.4-7.7s).
$models = Join-Path $PSScriptRoot 'models'
New-Item -ItemType Directory -Force $models | Out-Null
$files = @(
    @{ Name = 'en_US-lessac-medium.onnx';      Url = 'https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx' },
    @{ Name = 'en_US-lessac-medium.onnx.json'; Url = 'https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json' }
)
foreach ($f in $files) {
    $dest = Join-Path $models $f.Name
    if (Test-Path $dest) { Write-Host "  $($f.Name) already present"; continue }
    Write-Host "  downloading $($f.Name)..."
    try {
        Invoke-WebRequest -Uri $f.Url -OutFile $dest
    } catch {
        Write-Warning "Download failed for $($f.Name). Grab it manually from https://huggingface.co/rhasspy/piper-voices and place it in .\models\ — the smoke test skips TTS until then."
    }
}

# 3. .env
if (-not (Test-Path (Join-Path $PSScriptRoot '.env'))) {
    Copy-Item (Join-Path $PSScriptRoot '.env.example') (Join-Path $PSScriptRoot '.env')
    Write-Host "  created .env — paste your GROQ_API_KEY into it" -ForegroundColor Yellow
}

Write-Host "`nDone. Next:  uv run scripts/smoke_test.py" -ForegroundColor Green
