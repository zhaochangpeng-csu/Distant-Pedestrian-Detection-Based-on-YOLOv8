param([string]$Python = '', [switch]$SkipTorch)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot
$EnvRoot = Join-Path $ProjectRoot '.venv'
$VenvPython = Join-Path $EnvRoot 'Scripts\python.exe'
$CondaPython = Join-Path $EnvRoot 'python.exe'

function Test-Python311([string]$Executable, [string[]]$PrefixArgs) {
    try {
        & $Executable @PrefixArgs -c 'import sys;sys.exit(sys.version_info[:2]!=(3,11))' 2>$null | Out-Null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    }
}

if (Test-Path -LiteralPath $VenvPython) {
    $TaskPython = $VenvPython
} elseif (Test-Path -LiteralPath $CondaPython) {
    $TaskPython = $CondaPython
} else {
    $Candidates = @()
    if ($Python) {
        if ($Python -eq 'py') {
            $Candidates += @{ Executable = 'py'; Prefix = @('-3.11') }
        } else {
            $Candidates += @{ Executable = $Python; Prefix = @() }
        }
    } else {
        $Candidates += @{ Executable = 'py'; Prefix = @('-3.11') }
        $Candidates += @{ Executable = 'python3.11'; Prefix = @() }
        $Candidates += @{ Executable = 'python'; Prefix = @() }
    }
    $Creator = $null
    foreach ($Candidate in $Candidates) {
        if (-not (Get-Command $Candidate.Executable -ErrorAction SilentlyContinue)) { continue }
        $PrefixArgs = @($Candidate.Prefix)
        if (Test-Python311 $Candidate.Executable $PrefixArgs) {
            $Creator = $Candidate
            break
        }
    }
    if ($Creator) {
        $PrefixArgs = @($Creator.Prefix)
        Write-Host "使用 $($Creator.Executable) 创建项目 Python 3.11 虚拟环境。"
        & $Creator.Executable @PrefixArgs -m venv $EnvRoot
        if ($LASTEXITCODE -ne 0) { throw '创建项目虚拟环境失败。' }
        $TaskPython = $VenvPython
    } elseif ($Python) {
        throw "指定的 -Python '$Python' 不存在或不是 Python 3.11。请传入 Python 3.11 的 python.exe 完整路径。"
    } elseif (Get-Command conda -ErrorAction SilentlyContinue) {
        Write-Host '未找到 Python 3.11；使用 Conda 在项目 .venv 内创建独立 Python 3.11 环境。'
        & conda create --prefix $EnvRoot python=3.11 pip -y
        if ($LASTEXITCODE -ne 0) { throw 'Conda 创建 Python 3.11 环境失败。请检查 Conda 网络/频道配置。' }
        $TaskPython = $CondaPython
    } else {
        throw '未找到 Python 3.11、py 启动器或 Conda。请安装 Python 3.11，或用 -Python 指定 python.exe 完整路径。'
    }
}

if (-not (Test-Path -LiteralPath $TaskPython)) { throw "环境创建后仍找不到解释器：$TaskPython" }
if (-not (Test-Python311 -Executable $TaskPython -PrefixArgs @())) { throw "项目环境不是 Python 3.11：$TaskPython" }
Write-Host "项目解释器：$TaskPython"
$ReadyMarker = Join-Path $EnvRoot '.setup_complete'
if (Test-Path -LiteralPath $ReadyMarker) { Remove-Item -LiteralPath $ReadyMarker }
& $TaskPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'pip 升级失败。' }
if (-not $SkipTorch) {
    & $TaskPython -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE -ne 0) { throw 'PyTorch cu128 安装失败，请检查网络。' }
}
& $TaskPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw '项目依赖安装失败。' }
& $TaskPython run.py check --strict
if ($LASTEXITCODE -ne 0) { throw 'GPU 兼容性检查未通过，请查看上方错误与驱动版本。' }
[System.IO.File]::WriteAllText($ReadyMarker, 'Python 3.11 / PyTorch 2.7.1 / CUDA 12.8 verified')
Write-Host '环境准备完成。下一步运行 scripts\run_experiments.ps1。'
