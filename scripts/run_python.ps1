param([Parameter(ValueFromRemainingArguments = $true)][string[]]$PythonArgs)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot
$TaskPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $TaskPython)) {
    $TaskPython = Join-Path $ProjectRoot '.venv\python.exe'
}
if (-not (Test-Path -LiteralPath $TaskPython)) {
    throw '未找到项目 Python 环境。请先运行 scripts\setup_windows.ps1。'
}
& $TaskPython @PythonArgs
exit $LASTEXITCODE
