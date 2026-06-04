[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$RepoRoot = "",
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
    "-File", $scriptPath
)
if ($ExtraArgs) {
    $args += $ExtraArgs
}

Write-BuildStep "Building CUDA YOLO annotation worker through cuda\build-yolo-worker.ps1" "worker"
Invoke-External "powershell" $args -DryRun:$DryRun
Write-BuildStep "Done: CUDA worker build completed" "worker"
