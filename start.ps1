param([ValidateSet('tiny','base','phowhisper-base','phowhisper-vietsuper','phowhisper-tiny','phowhisper-tiny-lora','zipformer-vi-30m','zipformer-vi-30m-lora','zipformer-vi-30m-lora-low-lr20')][string]$Whisper = 'zipformer-vi-30m', [int]$Threads = 2, [int]$Port = 8000, [switch]$Demo)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) { throw 'Run prepare.ps1 first.' }
$env:EXAMWATCH_WHISPER = $Whisper
$env:EXAMWATCH_THREADS = [string][Math]::Max(1, [Math]::Min(8, $Threads))
$env:OMP_NUM_THREADS = $env:EXAMWATCH_THREADS
$env:OPENBLAS_NUM_THREADS = $env:EXAMWATCH_THREADS
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:MPLCONFIGDIR = Join-Path $PSScriptRoot '.tmp/matplotlib'
if ($Demo) { $env:EXAMWATCH_DEMO_MODE = '1' } else { Remove-Item Env:EXAMWATCH_DEMO_MODE -ErrorAction SilentlyContinue }
if ($Demo) { Write-Host 'Chế độ mô phỏng: dùng dữ liệu mẫu cục bộ, bỏ qua đăng nhập Firebase.' }
Write-Host "ExamWatch: http://localhost:$Port (Ctrl+C to stop)"
& .venv/Scripts/python.exe -m uvicorn examwatch.main:create_app --factory --host 127.0.0.1 --port $Port --workers 1
