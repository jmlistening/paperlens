$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSScriptRoot
$python = if ($env:PAPERLENS_PYTHON) { $env:PAPERLENS_PYTHON } else { "python" }
if (-not (Get-Command $python -ErrorAction SilentlyContinue) -and -not (Test-Path $python)) {
    throw "Python interpreter not found: $python. Set PAPERLENS_PYTHON or activate your venv."
}
Set-Location $project
& $python -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python -m unittest discover -s . -p "test_eval.py" -v
exit $LASTEXITCODE
