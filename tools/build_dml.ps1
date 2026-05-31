[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$RepoRoot = "",
    [ValidateSet("DML", "Debug", "Release")]
    [string]$Configuration = "DML",
    [string]$Platform = "x64",
    [string]$Project = "",
    [switch]$NonInteractive,
    [switch]$DryRun,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraMSBuildArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. "$PSScriptRoot\build_common.ps1"

if (-not [string]::IsNullOrWhiteSpace($RepoRoot)) {
    $script:RepoRootOverride = $RepoRoot
}

$repo = Get-RepoRoot
if ([string]::IsNullOrWhiteSpace($Project)) {
    $Project = Join-Path $repo "0BS_box_2.vcxproj"
}

if (-not (Test-Path -LiteralPath $Project -PathType Leaf)) {
    throw "DML project file was not found: $Project"
}

Write-BuildStep "Building 0BS DirectML ($Configuration|$Platform)" "dml"
Invoke-VisualStudioProjectBuild `
    -Project $Project `
    -Configuration $Configuration `
    -Platform $Platform `
    -AdditionalProperties $ExtraMSBuildArgs `
    -DryRun:$DryRun

Write-BuildStep "Done: x64\$Configuration\0BS.exe" "dml"
