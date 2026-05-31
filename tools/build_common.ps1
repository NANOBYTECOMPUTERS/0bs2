[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-RepoRoot {
    $override = Get-Variable -Name RepoRootOverride -Scope Script -ErrorAction SilentlyContinue
    if ($override -and -not [string]::IsNullOrWhiteSpace([string]$override.Value)) {
        return [System.IO.Path]::GetFullPath([string]$override.Value)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
}

function Resolve-RepoPath {
    param([Parameter(Mandatory)][string]$RelativePath)
    return [System.IO.Path]::GetFullPath((Join-Path (Get-RepoRoot) $RelativePath))
}

function Write-BuildStep {
    param(
        [Parameter(Mandatory)][string]$Message,
        [string]$Prefix = "build"
    )
    Write-Host "[$Prefix] $Message" -ForegroundColor Cyan
}

function Get-CommandPath {
    param([Parameter(Mandatory)][string]$Name)
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    return $null
}

function Get-VsWherePath {
    $fromPath = Get-CommandPath "vswhere.exe"
    if ($fromPath) { return $fromPath }

    $candidate = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path -LiteralPath $candidate) { return $candidate }
    return $null
}

function Get-VisualStudioInstallation {
    $vswhere = Get-VsWherePath
    if (-not $vswhere) {
        throw "vswhere.exe was not found. Install Visual Studio Build Tools or Visual Studio Community with C++ tools."
    }

    $installPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($installPath)) {
        throw "No Visual Studio installation with C++ tools was found."
    }

    $installPath = $installPath.Trim()
    $vsDevCmd = Join-Path $installPath "Common7\Tools\VsDevCmd.bat"
    if (-not (Test-Path -LiteralPath $vsDevCmd)) {
        throw "VsDevCmd.bat was not found: $vsDevCmd"
    }

    [pscustomobject]@{
        InstallationPath = $installPath
        VsDevCmd = $vsDevCmd
    }
}

function Import-VisualStudioEnvironment {
    param(
        [string]$Architecture = "x64",
        [string]$HostArchitecture = "x64"
    )

    $requiredTools = @("cl.exe", "link.exe", "rc.exe", "mt.exe")
    $missingTools = @($requiredTools | Where-Object { -not (Get-CommandPath $_) })
    if ($missingTools.Count -eq 0) {
        return
    }

    $vs = Get-VisualStudioInstallation
    Write-BuildStep "Importing Visual Studio environment from VsDevCmd.bat" "toolchain"
    $cmdLine = '"' + $vs.VsDevCmd + '"' + " -arch=$Architecture -host_arch=$HostArchitecture && set"
    $envLines = & cmd.exe /s /c $cmdLine
    if ($LASTEXITCODE -ne 0) {
        throw "VsDevCmd.bat failed to initialize the compiler environment."
    }

    foreach ($line in $envLines) {
        if ($line -match "^([^=]+)=(.*)$") {
            [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], "Process")
        }
    }

    $missingTools = @($requiredTools | Where-Object { -not (Get-CommandPath $_) })
    if ($missingTools.Count -gt 0) {
        throw "Visual Studio environment is incomplete. Missing: $($missingTools -join ', ')."
    }
}

function Resolve-MSBuild {
    $cmd = Get-Command MSBuild.exe -ErrorAction SilentlyContinue
    if ($cmd -and (Test-Path -LiteralPath $cmd.Source -PathType Leaf)) {
        return $cmd.Source
    }

    $vswhere = Get-VsWherePath
    if ($vswhere) {
        $found = & $vswhere -latest -products * -requires Microsoft.Component.MSBuild -find "MSBuild\Current\Bin\MSBuild.exe" |
            Select-Object -First 1
        if ($found -and (Test-Path -LiteralPath $found -PathType Leaf)) {
            return $found
        }
    }

    $vs = Get-VisualStudioInstallation
    $candidate = Join-Path $vs.InstallationPath "MSBuild\Current\Bin\MSBuild.exe"
    if (Test-Path -LiteralPath $candidate -PathType Leaf) {
        return $candidate
    }

    throw "MSBuild.exe was not found. Install Visual Studio Build Tools or run from a Developer PowerShell."
}

function Invoke-External {
    param(
        [Parameter(Mandatory)][string]$Exe,
        [Parameter(Mandatory)][string[]]$ArgumentList,
        [switch]$DryRun
    )

    $printable = $Exe + " " + (($ArgumentList | ForEach-Object {
        if ($_ -match "\s") { '"' + $_ + '"' } else { $_ }
    }) -join " ")
    Write-Host ">> $printable"

    if ($DryRun) { return }

    & $Exe @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Exe"
    }
}

function Invoke-VisualStudioProjectBuild {
    param(
        [Parameter(Mandatory)][string]$Project,
        [Parameter(Mandatory)][string]$Configuration,
        [string]$Platform = "x64",
        [string[]]$AdditionalProperties = @(),
        [switch]$DryRun
    )

    if ($null -eq $AdditionalProperties) {
        $AdditionalProperties = @()
    }

    Import-VisualStudioEnvironment
    $msbuild = Resolve-MSBuild
    $args = @(
        $Project,
        "/p:Configuration=$Configuration",
        "/p:Platform=$Platform"
    ) + $AdditionalProperties + @("/m", "/v:minimal")

    Invoke-External $msbuild $args -DryRun:$DryRun
}
