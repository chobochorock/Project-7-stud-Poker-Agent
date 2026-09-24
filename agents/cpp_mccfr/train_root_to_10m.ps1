param(
    [int]$Chunks = 99,
    [int]$RootsPerChunk = 100000,
    [int64]$InitialRoots = 100000,
    [uint64]$Seed = 9400,
    [int]$Ante = 1000,
    [uint64]$PowerCacheMax = 250000,
    [string]$Executable = ".\agents\cpp_mccfr\bin\stud_mccfr_bounded_cache.exe",
    [string]$Bucket = "power",
    [string]$Atlas = "agents\cpp_mccfr\data\power64_v1.bin",
    [string]$BaseModel = "agents\cpp_mccfr\data\root_mccfr_ante1000_100k.bin",
    [string]$OutputModel = "agents\cpp_mccfr\data\root_mccfr_ante1000_10m.bin"
)

$ErrorActionPreference = "Stop"
$repo = $PSScriptRoot
while (!(Test-Path -LiteralPath (Join-Path $repo 'PROJECT_LAYOUT.json'))) {
    $parent = Split-Path $repo -Parent
    if (!$parent -or $parent -eq $repo) { throw 'Project root not found' }
    $repo = $parent
}
Set-Location $repo

$model = $OutputModel
$next = "$model.next"
$progress = "$model.progress"
$stdout = "$model.stdout.log"
$stderr = "$model.stderr.log"
if (!(Test-Path -LiteralPath $model)) {
    Copy-Item -LiteralPath $BaseModel -Destination $model
    Set-Content -LiteralPath $progress -Value 0
} elseif (!(Test-Path -LiteralPath $progress)) {
    throw "existing output has no progress file: $model"
}

$completed = [int](Get-Content -LiteralPath $progress)
for ($chunk = $completed + 1; $chunk -le $Chunks; ++$chunk) {
    $savedErrorPreference = $ErrorActionPreference
    try {
        # Windows PowerShell 5.1 wraps native stderr progress as NativeCommandError.
        $ErrorActionPreference = "Continue"
        & $Executable `
            --bucket $Bucket `
            --load-atlas $Atlas `
            --start-street 5 `
            --algorithm mccfr `
            --ante $Ante `
            --load $model `
            --root-iterations $RootsPerChunk `
            --root-report-every $RootsPerChunk `
            --power-cache-max $PowerCacheMax `
            --hands 2 `
            --iterations 0 `
            --opponent heuristic `
            --seed ($Seed + $chunk) `
            --save $next `
            1>> $stdout `
            2>> $stderr
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $savedErrorPreference
    }
    if ($exitCode -ne 0) {
        throw "training chunk $chunk failed with exit code $exitCode"
    }
    Move-Item -LiteralPath $next -Destination $model -Force
    Set-Content -LiteralPath $progress -Value $chunk
        Add-Content -LiteralPath $stdout -Value (
        "completed chunk $chunk/$Chunks; total roots=" +
        ($InitialRoots + $chunk * $RootsPerChunk))
}
