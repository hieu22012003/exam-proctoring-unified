param([ValidateSet('tiny','base','phowhisper-base','zipformer-vi-30m')][string]$Whisper = 'zipformer-vi-30m')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:TEMP = Join-Path $PSScriptRoot '.tmp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Path $env:TEMP -Force | Out-Null
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    py -3.12 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.12 (64-bit) and retry.' }
}
if ($Whisper -eq 'zipformer-vi-30m') {
    & .venv/Scripts/python.exe -m pip install -r packaging/windows/requirements-portable.txt -c packaging/windows/constraints-portable.txt --only-binary=:all:
} else {
    & .venv/Scripts/python.exe -m pip install -r requirements.txt -c requirements-lock.txt --only-binary=:all:
}
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& .venv/Scripts/python.exe -m pip install 'ultralytics>=8.3.0,<9' 'firebase-admin>=6.5,<8'
if ($LASTEXITCODE -ne 0) { throw 'Phone detector or Firebase dependency installation failed.' }
& .venv/Scripts/python.exe scripts/prepare_models.py --whisper $Whisper
if ($LASTEXITCODE -ne 0) { throw 'Model preparation failed. See output above.' }
