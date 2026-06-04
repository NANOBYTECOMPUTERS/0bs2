[CmdletBinding(PositionalBinding = $false)]
param(
    [ValidateSet("DML", "CUDA", "WORKER", "ALL", "")]
    [string]$Backend = "",
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
    Write-Host "  .\build_no-options.bat"
    Write-Host "  powershell -ExecutionPolicy Bypass -File .\build_no-options.ps1 -Backend DML"
    Write-Host "  powershell -ExecutionPolicy Bypass -File .\build_no-options.ps1 -Backend CUDA"
    Write-Host ""
    Write-Host "Runs existing 0BS build paths without dependency prompts or setup steps."
    exit 0
}

if ([string]::IsNullOrWhiteSpace($Backend)) {
    $Backend = Select-BuildTarget -NonInteractive:$NonInteractive
}

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$builder = Join-Path $repoRoot "BUILDER.ps1"
$builderArgs = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $builder, "-Backend", $Backend, "-NonInteractive")
if ($DryRun) {
    $builderArgs += "-DryRun"
}
if ($BuildArgs) {
    $builderArgs += $BuildArgs
}
& powershell @builderArgs
exit $LASTEXITCODE
