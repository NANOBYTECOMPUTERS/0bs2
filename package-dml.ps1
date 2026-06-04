[CmdletBinding()]
param(
    [string]$SourceDir = "",
    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"

$scriptRoot = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($scriptRoot)) {
    $scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
}
if ([string]::IsNullOrWhiteSpace($scriptRoot)) {
    $scriptRoot = (Get-Location).Path
}

if ([string]::IsNullOrWhiteSpace($SourceDir)) {
    $SourceDir = Join-Path -Path $scriptRoot -ChildPath "x64\DML"
}
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path -Path $scriptRoot -ChildPath "dist\0BS"
}

function Get-FullPath {
    param([Parameter(Mandatory = $true)][string]$Path)

    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }

    return [System.IO.Path]::GetFullPath((Join-Path -Path (Get-Location) -ChildPath $Path))
}

function Copy-RequiredFile {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$From,
        [Parameter(Mandatory = $true)][string]$To
    )

    $sourcePath = Join-Path -Path $From -ChildPath $Name
    if (-not (Test-Path -LiteralPath $sourcePath -PathType Leaf)) {
        throw "Required package input '$Name' was not found in '$From'. Build the DML configuration first."
    }

    Copy-Item -LiteralPath $sourcePath -Destination (Join-Path -Path $To -ChildPath $Name) -Force
}

function Copy-OptionalMatches {
    param(
        [Parameter(Mandatory = $true)][string[]]$Patterns,
        [Parameter(Mandatory = $true)][string]$From,
        [Parameter(Mandatory = $true)][string]$To
    )

    foreach ($pattern in $Patterns) {
        Get-ChildItem -LiteralPath $From -Filter $pattern -File -ErrorAction SilentlyContinue |
            ForEach-Object {
                Copy-Item -LiteralPath $_.FullName -Destination (Join-Path -Path $To -ChildPath $_.Name) -Force
            }
    }
}

function Copy-DirectoryChildren {
    param(
        [Parameter(Mandatory = $true)][string]$From,
        [Parameter(Mandatory = $true)][string]$To
    )

    if (Test-Path -LiteralPath $From -PathType Container) {
        Get-ChildItem -LiteralPath $From -Force |
            ForEach-Object {
                Copy-Item -LiteralPath $_.FullName -Destination $To -Recurse -Force
            }
    }
}

function Ensure-Placeholder {
    param(
        [Parameter(Mandatory = $true)][string]$Directory,
        [Parameter(Mandatory = $true)][string]$Text
    )

    $content = @(Get-ChildItem -LiteralPath $Directory -Force -ErrorAction SilentlyContinue)
    if ($content.Count -eq 0) {
        Set-Content -LiteralPath (Join-Path -Path $Directory -ChildPath ".keep") -Value $Text -Encoding ASCII
    }
}

$sourcePath = Get-FullPath -Path $SourceDir
$outputPath = Get-FullPath -Path $OutputDir

if (-not (Test-Path -LiteralPath $sourcePath -PathType Container)) {
    throw "SourceDir '$sourcePath' does not exist."
}

if ($sourcePath.TrimEnd("\") -ieq $outputPath.TrimEnd("\")) {
    throw "OutputDir must be different from SourceDir."
}

if ($outputPath -eq [System.IO.Path]::GetPathRoot($outputPath)) {
    throw "Refusing to package into a drive root."
}

if (Test-Path -LiteralPath $outputPath) {
    Remove-Item -LiteralPath $outputPath -Recurse -Force
}

New-Item -ItemType Directory -Path $outputPath -Force | Out-Null

Copy-RequiredFile -Name "0BS.exe" -From $sourcePath -To $outputPath
# rzctl.dll is no longer required: Razer logic has been inlined directly into the executable
# (stealth improvement). It may still appear in some historical builds but is optional.
Copy-OptionalMatches -Patterns @(
    "ghub_mouse.dll",
    "opencv_*.dll",
    "onnxruntime*.dll",
    "DirectML*.dll"
) -From $sourcePath -To $outputPath

$configDir = Join-Path -Path $outputPath -ChildPath "config"
$modelsDir = Join-Path -Path $outputPath -ChildPath "models"
New-Item -ItemType Directory -Path $configDir -Force | Out-Null
New-Item -ItemType Directory -Path $modelsDir -Force | Out-Null

$sourceConfigDir = Join-Path -Path $sourcePath -ChildPath "config"
$sourceModelsDir = Join-Path -Path $sourcePath -ChildPath "models"
Copy-DirectoryChildren -From $sourceConfigDir -To $configDir
Copy-DirectoryChildren -From $sourceModelsDir -To $modelsDir

$sourceConfigFile = Join-Path -Path $sourcePath -ChildPath "config.ini"
if (Test-Path -LiteralPath $sourceConfigFile -PathType Leaf) {
    Copy-Item -LiteralPath $sourceConfigFile -Destination (Join-Path -Path $outputPath -ChildPath "config.ini") -Force
}

Ensure-Placeholder -Directory $configDir -Text "Place configuration files here."
Ensure-Placeholder -Directory $modelsDir -Text "Place model files here."

$modelFiles = @(
    Get-ChildItem -LiteralPath $modelsDir -File -Recurse -Force -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -ne ".keep" }
)
if ($modelFiles.Count -eq 0) {
    Write-Warning "No model files were found under '$modelsDir'."
}

Write-Host "Packaged 0BS to $outputPath"
