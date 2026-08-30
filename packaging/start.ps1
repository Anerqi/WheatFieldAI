# packaging/start.ps1
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$LogsDir = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $LogsDir "streamlit_$Timestamp.log"
$ErrLogFile = Join-Path $LogsDir "streamlit_$Timestamp.err.log"

$Port = 8501
if ($env:WHEATWEED_PORT) {
    if ($env:WHEATWEED_PORT -match '^\d+$') {
        $Port = [int]$env:WHEATWEED_PORT
    }
    else {
        Write-Host "[FAIL] WHEATWEED_PORT 不是有效端口：$($env:WHEATWEED_PORT)" -ForegroundColor Red
        pause
        exit 1
    }
}

if ($Port -lt 1024 -or $Port -gt 65535) {
    Write-Host "[FAIL] 端口必须位于 1024-65535：$Port" -ForegroundColor Red
    pause
    exit 1
}

$Url = "http://localhost:$Port"

Write-Host "============================================"
Write-Host " WheatFieldAI - Local Web Launcher"
Write-Host "============================================"
Write-Host "项目目录: $Root"
Write-Host "端口: $Port"
Write-Host "日志: $LogFile (stdout) / $ErrLogFile (stderr)"
Write-Host ""

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "[FAIL] 未找到 Python。" -ForegroundColor Red
    Write-Host "请安装 Python 3.13.x，并确保 python 已加入 PATH。"
    pause
    exit 1
}

Write-Host "[1/4] 检查 Python / 依赖 / PyTorch / CUDA..."
& python -B "$Root\packaging\check_env.py"
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "环境检查失败。" -ForegroundColor Red
    Write-Host "请查看上述输出并修复依赖；此阶段不会自动安装任何包。"
    pause
    exit 1
}

Write-Host ""
Write-Host "[2/4] 检查模型与配置资产..."
& python -B "$Root\packaging\check_models.py"
$ModelCode = $LASTEXITCODE

if ($ModelCode -eq 1) {
    Write-Host ""
    Write-Host "模型/配置存在结构错误，已停止启动。" -ForegroundColor Red
    Write-Host "请根据上方 [FAIL] 信息修复后重新启动。"
    pause
    exit 1
}

if ($ModelCode -eq 2) {
    Write-Host ""
    Write-Host "[WARN] 模型资产未完整配置。" -ForegroundColor Yellow
    Write-Host "应用仍会启动，但缺失模型的任务只显示「模型未配置」，不会生成模拟结果。"
}

Write-Host ""
Write-Host "[3/4] 检查端口 $Port..."

$portInUse = $false
try {
    $listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if ($listener) {
        $portInUse = $true
    }
}
catch {
    try {
        $test = Test-NetConnection -ComputerName "127.0.0.1" -Port $Port -WarningAction SilentlyContinue
        if ($test.TcpTestSucceeded) {
            $portInUse = $true
        }
    }
    catch {
    }
}

if ($portInUse) {
    Write-Host "[FAIL] 端口 $Port 已被占用。" -ForegroundColor Red
    Write-Host "处理方式："
    Write-Host "1. 关闭占用该端口的程序；或"
    Write-Host "2. 设置环境变量，例如： `$env:WHEATWEED_PORT='8502'；然后重新运行 start.bat。"
    pause
    exit 1
}

Write-Host "[ OK ] 端口 $Port 可用。"

Write-Host ""
Write-Host "[4/4] 启动 Streamlit，stdout 日志：$LogFile"
Write-Host "       stderr 日志：$ErrLogFile"

$arguments = @(
    "-B",
    "-m",
    "streamlit",
    "run",
    "$Root\app.py",
    "--server.port",
    "$Port",
    "--server.headless",
    "true"
)

$proc = Start-Process `
    -FilePath "python" `
    -ArgumentList $arguments `
    -WorkingDirectory $Root `
    -RedirectStandardOutput $LogFile `
    -RedirectStandardError $ErrLogFile `
    -PassThru

$ready = $false

for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 1

    if ($proc.HasExited) {
        break
    }

    try {
        $resp = Invoke-WebRequest `
            -Uri $Url `
            -UseBasicParsing `
            -TimeoutSec 2

        if ($resp.StatusCode -eq 200) {
            $ready = $true
            break
        }
    }
    catch {
    }
}

if ($ready) {
    Write-Host ""
    Write-Host "[ OK ] Streamlit 已就绪。" -ForegroundColor Green
    Write-Host "浏览器地址：$Url"
    Write-Host "日志文件：$LogFile / $ErrLogFile"
    Write-Host ""
    Start-Process $Url
}
else {
    Write-Host ""
    Write-Host "[FAIL] Streamlit 未在 30 秒内就绪。" -ForegroundColor Red
    Write-Host "请查看日志：$LogFile / $ErrLogFile"
    Write-Host "常见关键词：KeyError、CUDA out of memory、Address already in use"
    Write-Host ""
    if (-not $proc.HasExited) {
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    }
    pause
    exit 1
}

$proc.WaitForExit()
$ExitCode = $proc.ExitCode

Write-Host ""
Write-Host "Streamlit 已退出，退出码：$ExitCode"
Write-Host "日志文件：$LogFile / $ErrLogFile"

if ($ExitCode -ne 0) {
    pause
}

exit $ExitCode
