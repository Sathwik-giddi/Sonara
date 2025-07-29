# Sonara one-time setup. Run from E:\PA:  ./setup.ps1
$ErrorActionPreference = 'Stop'

Write-Host "== Sonara setup ==" -ForegroundColor Cyan

# 1. Python env (uv pins 3.12 per pyproject; downloads it if missing)
uv sync

# 2. Kokoro TTS model files (~330MB total), skipped if already present
$models = Join-Path $PSScriptRoot 'models'
New-Item -ItemType Directory -Force $models | Out-Null
$files = @(
    @{ Name = 'kokoro-v1.0.onnx'; Url = 'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx' },
    @{ Name = 'voices-v1.0.bin';  Url = 'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin' }
)
foreach ($f in $files) {
    $dest = Join-Path $models $f.Name
    if (Test-Path $dest) { Write-Host "  $($f.Name) already present"; continue }
    Write-Host "  downloading $($f.Name)..."
    try {
        Invoke-WebRequest -Uri $f.Url -OutFile $dest
    } catch {
        Write-Warning "Download failed for $($f.Name). Grab it manually from https://github.com/thewh1teagle/kokoro-onnx/releases and place it in .\models\ — the smoke test skips TTS until then."
    }
}

# 3. .env
if (-not (Test-Path (Join-Path $PSScriptRoot '.env'))) {
    Copy-Item (Join-Path $PSScriptRoot '.env.example') (Join-Path $PSScriptRoot '.env')
    Write-Host "  created .env — paste your GROQ_API_KEY into it" -ForegroundColor Yellow
}

Write-Host "`nDone. Next:  uv run scripts/smoke_test.py" -ForegroundColor Green
