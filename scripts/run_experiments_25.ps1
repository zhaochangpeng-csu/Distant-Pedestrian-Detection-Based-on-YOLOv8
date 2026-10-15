param([switch]$OfflineData, [switch]$SkipDownload, [switch]$SkipPrepare, [switch]$FinalTest)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot
$TaskPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $TaskPython)) { $TaskPython = Join-Path $ProjectRoot '.venv\python.exe' }
if (-not (Test-Path -LiteralPath $TaskPython)) { throw '未找到项目Python环境，请先完成setup_windows.ps1。' }
if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot '.venv\.setup_complete'))) {
    throw '项目依赖或GPU检查尚未完成，请先运行setup_windows.ps1。'
}
function Run-Step([string[]]$StepArgs) {
    & $TaskPython @StepArgs
    if ($LASTEXITCODE -ne 0) { throw "步骤失败: $StepArgs" }
}

Run-Step @('run.py','check','--strict')
if (-not $FinalTest) {
    if (-not (Test-Path -LiteralPath 'data\prepared\manifest.json')) {
        if (-not $SkipDownload) {
            if ($OfflineData) { Run-Step @('run.py','download','--offline') }
            else { Run-Step @('run.py','download') }
        }
        if (-not $SkipPrepare) { Run-Step @('run.py','prepare','--strict-counts') }
    }
    foreach ($Variant in @('baseline','eca','p2','eca_p2')) {
        if (Test-Path -LiteralPath "outputs/runs/${Variant}_e25") {
            throw "已有${Variant}_e25训练目录；为防止误用不完整权重，不能直接重跑此脚本。请先检查并断点恢复。"
        }
    }
    Run-Step @('run.py','preflight','--config','configs/train_25.yaml','--strict')
    Run-Step @('run.py','train-all','--config','configs/train_25.yaml','--tag','e25')
    foreach ($Variant in @('baseline','eca','p2','eca_p2')) {
        Run-Step @('run.py','evaluate','--weights',"outputs/runs/${Variant}_e25/weights/best.pt",'--split','val','--output',"outputs/evaluation/${Variant}_e25_val_960")
    }
    Run-Step @('run.py','analyze','--split','val','--tag','e25')
} else {
    foreach ($Variant in @('baseline','eca','p2','eca_p2')) {
        Run-Step @('run.py','evaluate','--weights',"outputs/runs/${Variant}_e25/weights/best.pt",'--split','test','--output',"outputs/evaluation/${Variant}_e25_test_960")
    }
}
Run-Step @('run.py','pack','--output','outputs/results_bundle_e25.zip')
Write-Host '25轮结果包：outputs\results_bundle_e25.zip'
