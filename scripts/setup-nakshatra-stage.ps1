param(
    [string]$ExternalRoot = "",
    [ValidateSet("Cpu", "Cuda", "Rocm", "Metal", "Vulkan")]
    [string]$Backend = "Cpu",
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
if ($env:MYCELLIOS_EXTERNAL_RUNTIME_RESEARCH -ne "1") {
    throw "Archived external-runtime research harness. Set MYCELLIOS_EXTERNAL_RUNTIME_RESEARCH=1 only for an isolated research run."
}
$workspacePath = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
if (-not $ExternalRoot) {
    $ExternalRoot = Join-Path $workspacePath "runtime\external\nakshatra-stage"
}
$externalPath = [System.IO.Path]::GetFullPath($ExternalRoot)
$nakshatraPath = Join-Path $externalPath "nakshatra"
$llamaPath = Join-Path $externalPath "llama.cpp"
$backendKey = $Backend.ToLowerInvariant()
$buildPath = Join-Path $llamaPath "build-gdlp-nakshatra-$backendKey"

$nakshatraRepository = "https://github.com/fthrvi/nakshatra.git"
$nakshatraCommit = "0c16119713396ec6052400f3eb049c5e7a66cd94"
$llamaRepository = "https://github.com/ggml-org/llama.cpp.git"
$llamaCommit = "c46583b86bed573c4ff30685dae59874f124e664"
$isWindowsHost = (
    [System.Environment]::OSVersion.Platform -eq [System.PlatformID]::Win32NT
)

function Invoke-GitChecked {
    param([string]$WorkingDirectory, [string[]]$Arguments)
    & git -C $WorkingDirectory @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "git failed in ${WorkingDirectory}: $($Arguments -join ' ')"
    }
}

function Test-GitCommand {
    param([string]$WorkingDirectory, [string[]]$Arguments)
    $previousErrorActionPreference = $ErrorActionPreference
    $hasNativePreference = Test-Path -LiteralPath Variable:PSNativeCommandUseErrorActionPreference
    if ($hasNativePreference) {
        $previousNativePreference = $PSNativeCommandUseErrorActionPreference
    }
    try {
        # A failed probe is data, not a script error. PowerShell 7 can otherwise
        # promote a non-zero native exit to NativeCommandError under Stop.
        $ErrorActionPreference = "Continue"
        if ($hasNativePreference) {
            $PSNativeCommandUseErrorActionPreference = $false
        }
        & git -C $WorkingDirectory @Arguments 2>$null
        return $LASTEXITCODE -eq 0
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
        if ($hasNativePreference) {
            $PSNativeCommandUseErrorActionPreference = $previousNativePreference
        }
    }
}

function Test-NativeCommandAvailable {
    param([string[]]$Names)
    foreach ($commandName in $Names) {
        if (Get-Command -Name $commandName -CommandType Application -ErrorAction SilentlyContinue) {
            return $true
        }
    }
    return $false
}

function Assert-NakshatraBuildPrerequisites {
    $missingRequirements = @()
    if (-not (Test-NativeCommandAvailable @("cmake.exe", "cmake"))) {
        $missingRequirements += "CMake (cmake)"
    }
    if (-not (Test-NativeCommandAvailable @("cl.exe", "clang-cl.exe", "clang++.exe", "g++.exe"))) {
        $missingRequirements += "a C/C++ compiler (cl, clang-cl, clang++, or g++)"
    }
    if (-not (Test-NativeCommandAvailable @("ninja.exe", "nmake.exe", "mingw32-make.exe", "make.exe", "msbuild.exe"))) {
        $missingRequirements += "a build generator (Ninja, NMake, MinGW Make, Make, or MSBuild)"
    }
    if ($missingRequirements.Count -gt 0) {
        throw (
            "cannot build the Nakshatra daemon; missing: " +
            ($missingRequirements -join "; ") +
            ". Install or enter a configured native toolchain shell, then rerun. " +
            "Use -SkipBuild only to prepare and verify the pinned source tree."
        )
    }
}

function Initialize-PinnedRepository {
    param(
        [string]$Repository,
        [string]$Destination,
        [string]$Commit,
        [switch]$AllowTrackedChanges
    )
    if (-not (Test-Path -LiteralPath (Join-Path $Destination ".git"))) {
        New-Item -ItemType Directory -Force -Path (Split-Path $Destination -Parent) | Out-Null
        & git clone --filter=blob:none $Repository $Destination
        if ($LASTEXITCODE -ne 0) {
            throw "failed to clone $Repository"
        }
    }
    $head = (& git -C $Destination rev-parse HEAD 2>$null)
    if ($LASTEXITCODE -ne 0 -or $head.Trim() -ne $Commit) {
        $trackedChanges = (& git -C $Destination status --short --untracked-files=no)
        if ($trackedChanges) {
            throw "cannot repin a repository with tracked changes: $Destination"
        }
        Invoke-GitChecked $Destination @("fetch", "origin", $Commit, "--depth", "1")
        Invoke-GitChecked $Destination @("checkout", "--detach", $Commit)
    }
    $actual = (& git -C $Destination rev-parse HEAD).Trim()
    if ($actual -ne $Commit) {
        throw "repository pin mismatch at ${Destination}: $actual"
    }
    if (-not $AllowTrackedChanges) {
        $trackedChanges = (& git -C $Destination status --short --untracked-files=no)
        if ($trackedChanges) {
            throw "pinned repository has tracked modifications: $Destination"
        }
    }
}

if (-not $SkipBuild) {
    Assert-NakshatraBuildPrerequisites
}

New-Item -ItemType Directory -Force -Path $externalPath | Out-Null
Initialize-PinnedRepository $nakshatraRepository $nakshatraPath $nakshatraCommit
Initialize-PinnedRepository $llamaRepository $llamaPath $llamaCommit -AllowTrackedChanges

$patchRoot = Join-Path $nakshatraPath "experiments\v0.0\m4_patches"
$patchNames = @(
    "llama-model.h.patch",
    "llama-model.cpp.patch",
    "llama-model-loader.cpp.patch",
    "llama-graph.cpp.patch",
    "models_llama.cpp.patch"
)
foreach ($patchName in $patchNames) {
    $patchPath = Join-Path $patchRoot $patchName
    if (Test-GitCommand $llamaPath @("apply", "--check", "-p4", $patchPath)) {
        Invoke-GitChecked $llamaPath @("apply", "-p4", $patchPath)
        continue
    }
    if (-not (Test-GitCommand $llamaPath @("apply", "--reverse", "--check", "-p4", $patchPath))) {
        throw "Nakshatra patch is neither applicable nor already applied: $patchName"
    }
}

$examplePath = Join-Path $llamaPath "examples\nakshatra-spike"
New-Item -ItemType Directory -Force -Path $examplePath | Out-Null
Copy-Item -LiteralPath (Join-Path $nakshatraPath "experiments\v0.0\worker_daemon.cpp") `
    -Destination (Join-Path $examplePath "worker_daemon.cpp") -Force
$targetWorkerSource = Join-Path $examplePath "worker_daemon.cpp"
if ($isWindowsHost) {
    # Windows CRT standard streams default to text mode. The stdio protocol is
    # arbitrary binary, so disable CR/LF and EOF-byte translation explicitly.
    $workerSource = [System.IO.File]::ReadAllText($targetWorkerSource)
    $includeAnchor = [regex]::new('#include <unistd\.h>\r?\n')
    $patchedWorkerSource = $includeAnchor.Replace(
        $workerSource,
        "#include <unistd.h>`n#ifdef _WIN32`n#include <fcntl.h>`n#include <io.h>`n#endif`n",
        1
    )
    if ($patchedWorkerSource -eq $workerSource) {
        throw "cannot add Windows binary-stdio includes to pinned worker daemon"
    }
    $mainAnchor = [regex]::new('int main\(int argc, char \*\* argv\) \{\r?\n')
    $binaryStdioSetup = @'
int main(int argc, char ** argv) {
#ifdef _WIN32
    if (_setmode(_fileno(stdin), _O_BINARY) == -1 ||
        _setmode(_fileno(stdout), _O_BINARY) == -1) {
        fprintf(stderr, "[daemon] failed to enable binary stdio\n");
        return 5;
    }
#endif
'@
    $binaryStdioSetup += "`n"
    $finalWorkerSource = $mainAnchor.Replace(
        $patchedWorkerSource,
        $binaryStdioSetup,
        1
    )
    if ($finalWorkerSource -eq $patchedWorkerSource) {
        throw "cannot add Windows binary-stdio setup to pinned worker daemon"
    }
    [System.IO.File]::WriteAllText($targetWorkerSource, $finalWorkerSource)
}
$targetShmHeader = Join-Path $examplePath "shm_ring.hpp"
if ($isWindowsHost) {
    # Upstream's optional shared-memory ring requires POSIX mmap/fcntl. The
    # GDLP adapter uses stdio only, so retain the compile-time API while making
    # any accidental attempt to enable SHM fail explicitly at daemon startup.
    $windowsShmStub = @'
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace nakshatra { namespace fabric {

class ShmRingError : public std::runtime_error {
public:
    explicit ShmRingError(const std::string & message)
        : std::runtime_error(message) {}
};

class ShmRing {
public:
    static ShmRing attach(const std::string &) {
        throw ShmRingError("shared-memory transport unsupported on Windows");
    }

    std::size_t capacity() const noexcept { return 0; }

    void read_message_blocking(std::vector<std::uint8_t> &) {
        throw ShmRingError("shared-memory transport unsupported on Windows");
    }

    void write_message_blocking(const std::uint8_t *, std::size_t) {
        throw ShmRingError("shared-memory transport unsupported on Windows");
    }
};

}} // namespace nakshatra::fabric
'@
    [System.IO.File]::WriteAllText(
        $targetShmHeader,
        $windowsShmStub + [Environment]::NewLine
    )
}
else {
    Copy-Item -LiteralPath (Join-Path $nakshatraPath "experiments\v0.0\shm_ring.hpp") `
        -Destination $targetShmHeader -Force
}
$targetCmake = @'
set(TARGET llama-nakshatra-worker)
add_executable(${TARGET} worker_daemon.cpp)
install(TARGETS ${TARGET} RUNTIME)
if(MINGW)
    # The pinned static ggml-cpu archive references OpenMP. Put all static
    # llama/ggml archives and libgomp in one rescan group so GOMP_* and omp_*
    # resolve after ggml-cpu regardless of CMake's transitive-link ordering.
    set(NAKSHATRA_GGML_LINK_TARGETS common llama ggml ggml-cpu ggml-base)
    if(TARGET ggml-vulkan)
        list(APPEND NAKSHATRA_GGML_LINK_TARGETS ggml-vulkan)
    endif()
    target_link_libraries(${TARGET} PRIVATE
        "-Wl,--start-group"
        ${NAKSHATRA_GGML_LINK_TARGETS} gomp
        "-Wl,--end-group"
        ${CMAKE_THREAD_LIBS_INIT}
    )
else()
    target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
endif()
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_compile_definitions(${TARGET} PRIVATE
    NAKSHATRA_FABRIC_SHA="__NAKSHATRA_COMMIT__"
    NAKSHATRA_FABRIC_BUILD_HOST="gdlp-pinned-setup"
)
'@
$targetCmake = $targetCmake.Replace("__NAKSHATRA_COMMIT__", $nakshatraCommit)
[System.IO.File]::WriteAllText(
    (Join-Path $examplePath "CMakeLists.txt"),
    $targetCmake + [Environment]::NewLine
)

$examplesCmake = Join-Path $llamaPath "examples\CMakeLists.txt"
$cmakeText = [System.IO.File]::ReadAllText($examplesCmake)
$subdirectoryLine = "add_subdirectory(nakshatra-spike)"
if (-not $cmakeText.Contains($subdirectoryLine)) {
    [System.IO.File]::AppendAllText(
        $examplesCmake,
        [Environment]::NewLine + $subdirectoryLine + [Environment]::NewLine
    )
}

$allowedTrackedChanges = @(
    "examples/CMakeLists.txt",
    "src/llama-graph.cpp",
    "src/llama-model-loader.cpp",
    "src/llama-model.cpp",
    "src/llama-model.h",
    "src/models/llama.cpp"
)
$actualTrackedChanges = @(& git -C $llamaPath diff --name-only)
foreach ($changedPath in $actualTrackedChanges) {
    if ($changedPath -notin $allowedTrackedChanges) {
        throw "unexpected tracked change in pinned llama.cpp: $changedPath"
    }
}

if (-not $SkipBuild) {
    $backendArguments = switch ($Backend) {
        "Cuda" { @("-DGGML_CUDA=ON") }
        "Rocm" { @("-DGGML_HIPBLAS=ON") }
        "Metal" { @("-DGGML_METAL=ON") }
        "Vulkan" { @("-DGGML_VULKAN=ON") }
        default { @("-DGGML_METAL=OFF", "-DGGML_CUDA=OFF", "-DGGML_HIPBLAS=OFF", "-DGGML_VULKAN=OFF") }
    }
    & cmake -S $llamaPath -B $buildPath @backendArguments `
        "-DNAKSHATRA_FABRIC_SHA=$nakshatraCommit"
    if ($LASTEXITCODE -ne 0) {
        throw "Nakshatra llama.cpp configure failed"
    }
    & cmake --build $buildPath --target llama-nakshatra-worker --parallel
    if ($LASTEXITCODE -ne 0) {
        throw "Nakshatra daemon build failed"
    }
}

$binary = Get-ChildItem -LiteralPath $buildPath -Recurse -File -ErrorAction SilentlyContinue |
    Where-Object { $_.BaseName -eq "llama-nakshatra-worker" } |
    Select-Object -First 1
[pscustomobject]@{
    nakshatraRepository = $nakshatraRepository
    nakshatraCommit = $nakshatraCommit
    llamaRepository = $llamaRepository
    llamaCommit = $llamaCommit
    backend = $backendKey
    buildPath = $buildPath
    nakshatraPath = $nakshatraPath
    llamaPath = $llamaPath
    daemonBinary = if ($binary) { $binary.FullName } else { $null }
    sharedMemoryTransport = if ($isWindowsHost) { "unsupported" } else { "posix" }
    slicer = Join-Path $nakshatraPath "experiments\v0.0\partial_gguf.py"
} | ConvertTo-Json
