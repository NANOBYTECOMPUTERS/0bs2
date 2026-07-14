param(
    [string]$Configuration = "CUDA",
    [string]$Platform = "x64",
    [string]$Project = "$PSScriptRoot\0BS_cuda.vcxproj",
    [string]$CudaVersion = "13.3",
    [string]$CudaToolkitDir = $env:CudaToolkitDir,
    [string]$TensorRTDir = $env:TensorRTDir,
    [string]$OpenCVDir = $env:OpenCVDir
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "build-deps.ps1")

function Get-VsWherePath {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path -LiteralPath $vswhere -PathType Leaf) {
        return $vswhere
    }
    return $null
}

function Get-VisualStudioInstallRoots {
    $roots = @()
    $vswhere = Get-VsWherePath
    if ($vswhere) {
        $roots += & $vswhere -products * -property installationPath
    }

    $visualStudioRoot = Join-Path $env:ProgramFiles "Microsoft Visual Studio"
    if (Test-Path -LiteralPath $visualStudioRoot) {
        $roots += Get-ChildItem -Path $visualStudioRoot -Directory -ErrorAction SilentlyContinue |
            ForEach-Object {
                Get-ChildItem -Path $_.FullName -Directory -ErrorAction SilentlyContinue
            } |
            ForEach-Object FullName
    }

    $roots |
        Where-Object { ![string]::IsNullOrWhiteSpace($_) -and (Test-Path -LiteralPath $_) } |
        Select-Object -Unique
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

    foreach ($installRoot in Get-VisualStudioInstallRoots) {
        $candidate = Join-Path $installRoot "MSBuild\Current\Bin\MSBuild.exe"
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return $candidate
        }
    }

    throw "MSBuild.exe was not found. Open a Developer PowerShell or install Visual Studio Build Tools."
}

function Find-CudaPropsUnder {
    param([string]$Root)

    if ([string]::IsNullOrWhiteSpace($Root) -or !(Test-Path -LiteralPath $Root)) {
        return $null
    }

    $match = Get-ChildItem -Path $Root -Recurse -Filter "CUDA $CudaVersion.props" -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($match) {
        return $match.FullName
    }
    return $null
}

function Resolve-CudaProps {
    if (![string]::IsNullOrWhiteSpace($env:VCTargetsPath)) {
        $candidate = Join-Path $env:VCTargetsPath "BuildCustomizations\CUDA $CudaVersion.props"
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return $candidate
        }
    }

    foreach ($installRoot in Get-VisualStudioInstallRoots) {
        $candidate = Find-CudaPropsUnder -Root (Join-Path $installRoot "MSBuild\Microsoft\VC")
        if ($candidate) {
            return $candidate
        }
    }

    throw "CUDA $CudaVersion Visual Studio build customizations were not found. Install CUDA build customizations or run from a Developer PowerShell with VCTargetsPath set."
}

function Test-CudaToolkitDir {
    param([string]$Path)

    if ([string]::IsNullOrWhiteSpace($Path) -or !(Test-Path -LiteralPath $Path -PathType Container)) {
        return $false
    }

    return (Test-Path -LiteralPath (Join-Path $Path "include\cuda_runtime.h") -PathType Leaf) -and
        (Test-Path -LiteralPath (Join-Path $Path "bin\nvcc.exe") -PathType Leaf) -and
        (Test-Path -LiteralPath (Join-Path $Path "lib\x64\cudart.lib") -PathType Leaf)
}

function Resolve-CudaToolkitDir {
    if (![string]::IsNullOrWhiteSpace($CudaToolkitDir)) {
        $resolved = [System.IO.Path]::GetFullPath($CudaToolkitDir)
        if (Test-CudaToolkitDir -Path $resolved) {
            return $resolved
        }
        throw "CUDA Toolkit directory is incomplete: $CudaToolkitDir"
    }

    $versionedEnvName = "CUDA_PATH_V$($CudaVersion -replace '\.', '_')"
    $candidates = @(
        [Environment]::GetEnvironmentVariable($versionedEnvName, "Process"),
        [Environment]::GetEnvironmentVariable($versionedEnvName, "Machine"),
        (Join-Path $env:ProgramFiles "NVIDIA GPU Computing Toolkit\CUDA\v$CudaVersion"),
        $env:CUDA_PATH
    )

    $cudaRoot = Join-Path $env:ProgramFiles "NVIDIA GPU Computing Toolkit\CUDA"
    if (Test-Path -LiteralPath $cudaRoot -PathType Container) {
        $candidates += Get-ChildItem -Path $cudaRoot -Directory -Filter "v*" -ErrorAction SilentlyContinue |
            Sort-Object @{ Expression = { Get-VersionFromModuleName -Name $_.Name -Prefix "v" }; Descending = $true } |
            ForEach-Object FullName
    }

    foreach ($candidate in ($candidates | Where-Object { $_ } | Select-Object -Unique)) {
        if (Test-CudaToolkitDir -Path $candidate) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }

    throw "A complete CUDA Toolkit directory was not found. Install CUDA Toolkit or pass -CudaToolkitDir."
}

$msbuild = Resolve-MSBuild
$cudaProps = Resolve-CudaProps
$resolvedCudaToolkitDir = Resolve-CudaToolkitDir
$vcTargetsPath = Split-Path -Path (Split-Path -Path $cudaProps -Parent) -Parent
# Use forward slashes so a trailing separator cannot escape the quote boundary
# when PowerShell invokes native MSBuild with a path containing spaces.
$vcTargetsPathForMsBuild = ($vcTargetsPath -replace '\\', '/')
if (!$vcTargetsPathForMsBuild.EndsWith('/')) {
    $vcTargetsPathForMsBuild += '/'
}
$cudaToolkitDirForMsBuild = ($resolvedCudaToolkitDir -replace '\\', '/')
if (!$cudaToolkitDirForMsBuild.EndsWith('/')) {
    $cudaToolkitDirForMsBuild += '/'
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$tensorRT = Resolve-TensorRTDependency -RepoRoot $repoRoot -TensorRTDir $TensorRTDir -RequirePlugin
$opencv = Resolve-OpenCVDependency -RepoRoot $repoRoot -OpenCVDir $OpenCVDir -RequireCuda

$TensorRTDir = $tensorRT.Dir
$OpenCVDir = $opencv.Dir

Write-Host "TensorRT: $TensorRTDir (major $($tensorRT.Major))"
Write-Host "TensorRT libs: $($tensorRT.Libraries)"
Write-Host "OpenCV:   $OpenCVDir ($($opencv.WorldLib))"
Write-Host "CUDA:     $resolvedCudaToolkitDir (customizations $CudaVersion)"

$properties = @(
    "/p:Configuration=$Configuration",
    "/p:Platform=$Platform",
    "/p:CudaBuildCustomizationVersion=$CudaVersion",
    "/p:CudaToolkitDir=$cudaToolkitDirForMsBuild",
    "/p:CudaToolkitCustomDir=$cudaToolkitDirForMsBuild",
    "/p:TensorRTDir=$TensorRTDir",
    "/p:TensorRTCoreLib=$($tensorRT.CoreLib)",
    "/p:TensorRTOnnxParserLib=$($tensorRT.OnnxParserLib)",
    "/p:TensorRTPluginLib=$($tensorRT.PluginLib)",
    "/p:OpenCVDir=$OpenCVDir",
    "/p:OpenCVWorldLib=$($opencv.WorldLib)",
    "/p:OpenCVWorldDll=$($opencv.WorldDll)",
    "/p:VCTargetsPath=$vcTargetsPathForMsBuild"
)

& $msbuild $Project @properties /m /v:minimal
exit $LASTEXITCODE
