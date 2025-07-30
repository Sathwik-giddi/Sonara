# Mirror the Sonara vault into the global Second Brain.
#
#   Personal vault (authoritative): E:\PA\Sonara Intelligence
#   Global copy (for future model training): E:\Second Brain\Sonara Intelligence
#
# One-way: personal -> global. Edit the personal vault; never the copy.
# Each vault keeps its own .obsidian workspace settings (excluded from the mirror).
#
# Run:  ./scripts/sync-vaults.ps1

$ErrorActionPreference = 'Stop'

$source = 'E:\PA\Sonara Intelligence'
$target = 'E:\Second Brain\Sonara Intelligence'

if (-not (Test-Path $source)) { throw "Personal vault not found: $source" }
New-Item -ItemType Directory -Force $target | Out-Null

# /MIR mirrors (including deletions) so the copy never drifts.
# /XD .obsidian keeps each vault's own UI state.
robocopy $source $target /MIR /XD '.obsidian' /NFL /NDL /NJH /NJS /NP | Out-Null

# robocopy exit codes 0-7 are success; 8+ is a real failure.
if ($LASTEXITCODE -ge 8) { throw "robocopy failed with exit code $LASTEXITCODE" }

$count = (Get-ChildItem $target -Recurse -File -Filter *.md).Count
Write-Host "Synced $count notes -> $target" -ForegroundColor Green
