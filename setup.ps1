# Create an isolated Python environment and install the pinned dependencies.
# Dataset acquisition and training remain separate, explicit steps in README.md.
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
python -m venv .venv
if ($LASTEXITCODE -ne 0) { throw 'Python 3.13 is required. Install Python and retry.' }
& '.venv/Scripts/python.exe' -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
Write-Output 'Setup complete. Run ./run.ps1 to open PumpWatch.'
