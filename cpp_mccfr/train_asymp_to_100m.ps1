param(
    [int]$Chunks = 900,
    [int]$RootsPerChunk = 50000,
    [int64]$InitialEquivalentRoots = 10000000,
    [uint64]$Seed = 20000,
    [int]$Ante = 1000,
    [double]$Step = 0.0005,
    [double]$Mu = 0.01,
    [string]$BaseModel = "cpp_mccfr\root_mccfr_ante1000_10m.bin",
    [string]$OutputModel = "cpp_mccfr\root_asymp_ante1000_100m.bin"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent
Set-Location $repo

$next = "$OutputModel.next"
$progress = "$OutputModel.progress"
$stdout = "$OutputModel.stdout.log"
$stderr = "$OutputModel.stderr.log"
if (!(Test-Path -LiteralPath $OutputModel)) {
    if (!(Test-Path -LiteralPath $progress)) {
        Set-Content -LiteralPath $progress -Value 0
    }
} elseif (!(Test-Path -LiteralPath $progress)) {
    throw "existing output has no progress file: $OutputModel"
}

$completed = [int](Get-Content -LiteralPath $progress)
for ($chunk = $completed + 1; $chunk -le $Chunks; ++$chunk) {
    $modelArgs = if (Test-Path -LiteralPath $OutputModel) {
        @("--load", $OutputModel)
    } else {
        @("--init-from", $BaseModel)
    }
    $savedErrorPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & .\cpp_mccfr\stud_mccfr_asymp.exe `
            --bucket power `
            --load-atlas cpp_mccfr\power64_v1.bin `
            --start-street 5 `
            --algorithm asymp `
            --asymp-step $Step `
            --asymp-mu $Mu `
            --ante $Ante `
            @modelArgs `
            --root-iterations $RootsPerChunk `
            --root-report-every $RootsPerChunk `
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
        throw "AsymP training chunk $chunk failed with exit code $exitCode"
    }
    Move-Item -LiteralPath $next -Destination $OutputModel -Force
    Set-Content -LiteralPath $progress -Value $chunk
    Add-Content -LiteralPath $stdout -Value (
        "completed chunk $chunk/$Chunks; equivalent MCCFR roots=" +
        ($InitialEquivalentRoots + 2 * $chunk * $RootsPerChunk))
}
