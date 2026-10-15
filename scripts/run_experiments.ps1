param([switch]$SkipDownload, [switch]$OfflineData, [switch]$SkipPrepare, [switch]$FinalTest)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot
$TaskPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $TaskPython)) {
    $TaskPython = Join-Path $ProjectRoot '.venv\python.exe'
}
if (-not (Test-Path -LiteralPath $TaskPython)) {
    throw '未找到项目 Python 环境。请先运行 scripts\setup_windows.ps1；上一轮因缺少 py 命令未创建环境。'
}
if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot '.venv\.setup_complete'))) {
    throw '项目环境已创建，但依赖安装和GPU检查尚未完成。请重新运行 scripts\setup_windows.ps1，成功后再训练。'
}
function Run-Step([string[]]$StepArgs) {
    & $TaskPython @StepArgs
    if ($LASTEXITCODE -ne 0) { throw "步骤失败: $StepArgs" }
}
Run-Step @('run.py','check','--strict')
if (-not $SkipDownload) {
    if ($OfflineData) { Run-Step @('run.py','download','--offline') }
    else { Run-Step @('run.py','download') }
}
if (-not $SkipPrepare -and -not (Test-Path 'data\prepared\manifest.json')) { Run-Step @('run.py','prepare','--strict-counts') }
if (-not $FinalTest) {
    Run-Step @('run.py','preflight','--strict')
    Run-Step @('run.py','train-all')
    foreach ($Variant in @('baseline','eca','p2','eca_p2')) {
        Run-Step @('run.py','evaluate','--weights',"outputs/runs/$Variant/weights/best.pt",'--split','val','--output',"outputs/evaluation/${Variant}_val_960")
    }
    Run-Step @('run.py','analyze','--split','val')
} else {
    # 参数在验证集确定后，再单独运行 -FinalTest。测试数据不参与调参。
    foreach ($Variant in @('baseline','eca','p2','eca_p2')) {
        Run-Step @('run.py','evaluate','--weights',"outputs/runs/$Variant/weights/best.pt",'--split','test','--output',"outputs/evaluation/${Variant}_test_960")
    }
}
Run-Step @('run.py','pack')
Write-Host '将 outputs\results_bundle.zip 返回用于制作正式报告与答辩材料。'
