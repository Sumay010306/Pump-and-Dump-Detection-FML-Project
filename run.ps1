# Start the local historical-evaluation dashboard from this project directory.
# Prefer the project's virtual environment when one has been created.
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$projectPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) { $projectPython = 'python' }
& $projectPython -m streamlit run app.py --server.address 127.0.0.1
