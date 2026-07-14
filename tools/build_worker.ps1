[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$RepoRoot = "",
    [string]$CudaVersion = "13.3",
    [string]$CudaToolkitDir = "",
    [string]$TensorRTDir = "",
    [string]$OpenCVDir = "",
    [switch]$NonInteractive,
    [switch]$DryRun,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. "$PSScriptRoot\build_common.ps1"

if (-not [string]::IsNullOrWhiteSpace($RepoRoot)) {
    $script:RepoRootOverride = $RepoRoot
}

$repo = Get-RepoRoot
$scriptPath = Join-Path $repo "cuda\build-yolo-worker.ps1"
if (-not (Test-Path -LiteralPath $scriptPath -PathType Leaf)) {
    throw "Worker build script was not found: $scriptPath"
}

$args = @(
    "-NoProfile", "-ExecutionPolicy", "Bypass",
    "-File", $scriptPath,
    "-CudaVersion", $CudaVersion
)
if (-not [string]::IsNullOrWhiteSpace($TensorRTDir)) {
    $args += @("-TensorRTDir", $TensorRTDir)
}
if (-not [string]::IsNullOrWhiteSpace($CudaToolkitDir)) {
    $args += @("-CudaToolkitDir", $CudaToolkitDir)
}
if (-not [string]::IsNullOrWhiteSpace($OpenCVDir)) {
    $args += @("-OpenCVDir", $OpenCVDir)
}
if ($ExtraArgs) {
    $args += $ExtraArgs
}

Write-BuildStep "Building CUDA YOLO annotation worker through cuda\build-yolo-worker.ps1" "worker"
Invoke-External "powershell" $args -DryRun:$DryRun
Write-BuildStep "Done: CUDA worker build completed" "worker"
