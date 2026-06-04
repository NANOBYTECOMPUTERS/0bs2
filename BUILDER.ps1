[CmdletBinding(PositionalBinding = $false)]
param(
    [ValidateSet("DML", "CUDA", "WORKER", "ALL", "")]
    [string]$Backend = "",
    [ValidateSet("DML", "CUDA", "Debug", "Release", "")]
    [string]$Configuration = "",
    [string]$Platform = "x64",
    [string]$CudaVersion = "13.2",
    [string]$TensorRTDir = "",
    [switch]$Help,
    [switch]$NonInteractive,
    [switch]$DryRun,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$BuildArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Select-BuildTarget {
    param([switch]$NonInteractive)

    if ($NonInteractive) {
        throw "Backend is required in non-interactive mode. Use -Backend DML, CUDA, WORKER, or ALL."
    }

    Write-Host "Select build target"
    Write-Host "  1) DML"
    Write-Host "  2) CUDA"
    Write-Host "  3) WORKER"
    Write-Host "  4) ALL"

    while ($true) {
        $choice = Read-Host "Choose build target"
        if ($null -eq $choice) {
            throw "No build target was selected."
        }
        switch -Regex ($choice.Trim()) {
            "^(1|d|dml)$" { return "DML" }
            "^(2|c|cuda)$" { return "CUDA" }
            "^(3|w|worker)$" { return "WORKER" }
            "^(4|a|all)$" { return "ALL" }
        }
        Write-Host "Please enter DML, CUDA, WORKER, ALL, or 1-4."
    }
}

if ($Help -or ($BuildArgs -contains "--help") -or ($BuildArgs -contains "/?")) {
    Write-Host "Usage:"
    Write-Host "  .\BUILDER.bat"
    Write-Host "  powershell -ExecutionPolicy Bypass -File .\BUILDER.ps1 -Backend DML"
    Write-Host "  powershell -ExecutionPolicy Bypass -File .\BUILDER.ps1 -Backend CUDA -TensorRTDir C:\path\TensorRT"
    Write-Host "  powershell -ExecutionPolicy Bypass -File .\BUILDER.ps1 -Backend ALL -NonInteractive"
    Write-Host ""
    Write-Host "Targets reuse existing 0BS Visual Studio/MSBuild scripts. OpenCV is not rebuilt by this launcher."
    exit 0
}

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($Backend)) {
    $Backend = Select-BuildTarget -NonInteractive:$NonInteractive
}

function Invoke-BackendBuild {
    param([Parameter(Mandatory)][string]$Target)

    $scriptName = switch ($Target) {
        "DML" { "tools\build_dml.ps1" }
        "CUDA" { "tools\build_cuda.ps1" }
        "WORKER" { "tools\build_worker.ps1" }
        default { throw "Unsupported build target: $Target" }
    }

    $scriptPath = Join-Path $repoRoot $scriptName
    if (-not (Test-Path -LiteralPath $scriptPath -PathType Leaf)) {
        throw "Build script not found: $scriptPath"
    }

    $forwardedArgs = @()
    if ($NonInteractive) { $forwardedArgs += "-NonInteractive" }
    if ($DryRun) { $forwardedArgs += "-DryRun" }
    if ($Target -in @("DML", "CUDA") -and -not [string]::IsNullOrWhiteSpace($Configuration)) {
        $forwardedArgs += @("-Configuration", $Configuration)
    }
    if ($Target -in @("DML", "CUDA") -and -not [string]::IsNullOrWhiteSpace($Platform)) {
        $forwardedArgs += @("-Platform", $Platform)
    }
    if ($Target -eq "CUDA") {
        if (-not [string]::IsNullOrWhiteSpace($CudaVersion)) {
            $forwardedArgs += @("-CudaVersion", $CudaVersion)
        }
        if (-not [string]::IsNullOrWhiteSpace($TensorRTDir)) {
            $forwardedArgs += @("-TensorRTDir", $TensorRTDir)
        }
    }

    Write-Host "[BUILDER] Running $Target build..."
    & powershell -NoProfile -ExecutionPolicy Bypass -File $scriptPath @forwardedArgs @BuildArgs
    if ($LASTEXITCODE -ne 0) {
        throw "$Target build failed with exit code $LASTEXITCODE."
    }
}

if ($Backend -eq "ALL") {
    Invoke-BackendBuild "DML"
    Invoke-BackendBuild "CUDA"
    Invoke-BackendBuild "WORKER"
}
else {
    Invoke-BackendBuild $Backend
}
