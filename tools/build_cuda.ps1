[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$RepoRoot = "",
    [string]$Configuration = "CUDA",
    [string]$Platform = "x64",
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
$scriptPath = Join-Path $repo "cuda\build-cuda.ps1"
if (-not (Test-Path -LiteralPath $scriptPath -PathType Leaf)) {
    throw "CUDA build script was not found: $scriptPath"
}

$args = @(
    "-NoProfile", "-ExecutionPolicy", "Bypass",
    "-File", $scriptPath,
    "-Configuration", $Configuration,
    "-Platform", $Platform,
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

Write-BuildStep "Building 0BS CUDA through cuda\build-cuda.ps1" "cuda"
Invoke-External "powershell" $args -DryRun:$DryRun
Write-BuildStep "Done: x64\$Configuration\0BS.exe" "cuda"
