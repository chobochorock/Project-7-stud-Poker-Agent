param(
    [int]$Workers = 4,
    [uint64]$TargetRoots = 100000000,
    [uint64]$RootsPerWorker = 100000,
    [uint64]$Seed = 10000,
    [int]$Ante = 1000,
    [uint64]$BaseRoots = 10000000,
    [string]$BaseModel = "cpp_mccfr\root_mccfr_ante1000_10m.bin",
    [string]$OutputModel = "cpp_mccfr\root_mccfr_ante1000_100m.bin"
)

$ErrorActionPreference = "Stop"
if ($Workers -lt 1 -or $RootsPerWorker -lt 1 -or $TargetRoots -le $BaseRoots) {
    throw "Workers and RootsPerWorker must be positive; TargetRoots must exceed BaseRoots."
}

$repo = Split-Path $PSScriptRoot -Parent
Set-Location $repo
$exe = Join-Path $repo "cpp_mccfr\stud_mccfr.exe"
$progress = "$OutputModel.parallel.progress"
$next = "$OutputModel.next"

if (!(Test-Path -LiteralPath $OutputModel)) {
    Copy-Item -LiteralPath $BaseModel -Destination $OutputModel
    Set-Content -LiteralPath $progress -Value $BaseRoots
} elseif (!(Test-Path -LiteralPath $progress)) {
    throw "existing output has no progress file: $OutputModel"
}

$completed = [uint64](Get-Content -LiteralPath $progress)
$round = 0
while ($completed -lt $TargetRoots) {
    ++$round
    $roundStarted = Get-Date
    $roundRemaining = [uint64][Math]::Min(
        [double]($TargetRoots - $completed),
        [double]($Workers * $RootsPerWorker))
    $workersThisRound = @()

    for ($worker = 0; $worker -lt $Workers -and $roundRemaining -gt 0; ++$worker) {
        $workerRoots = [uint64][Math]::Min(
            [double]$RootsPerWorker,
            [double]$roundRemaining)
        $roundRemaining -= $workerRoots
        $workerModel = "$OutputModel.worker$worker.bin"
        $stdout = "$OutputModel.worker$worker.stdout.log"
        $stderr = "$OutputModel.worker$worker.stderr.log"
        $arguments = @(
            "--bucket", "power",
            "--load-atlas", "cpp_mccfr\power64_v1.bin",
            "--start-street", "5",
            "--algorithm", "mccfr",
            "--ante", "$Ante",
            "--load", "$OutputModel",
            "--root-iterations", "$workerRoots",
            "--root-report-every", "$workerRoots",
            "--hands", "2",
            "--iterations", "0",
            "--opponent", "heuristic",
            "--seed", "$($Seed + $round * $Workers + $worker)",
            "--save", "$workerModel"
        )
        $job = Start-Job -ScriptBlock {
            param($WorkingDirectory, $Executable, $Arguments, $Stdout, $Stderr)
            Set-Location $WorkingDirectory
            & $Executable @Arguments 1> $Stdout 2> $Stderr
            if ($LASTEXITCODE -ne 0) {
                throw "worker exited with code $LASTEXITCODE"
            }
        } -ArgumentList $repo, $exe, $arguments, $stdout, $stderr
        $workersThisRound += [pscustomobject]@{
            Job = $job
            Model = $workerModel
            Stdout = $stdout
            Stderr = $stderr
            Roots = $workerRoots
        }
    }

    $workersThisRound.Job | Wait-Job | Out-Null
    $failed = @($workersThisRound | Where-Object { $_.Job.State -ne "Completed" })
    if ($failed.Count) {
        $failed.Job | Receive-Job
        throw "parallel training round $round failed; worker logs were preserved"
    }
    $workersThisRound.Job | Receive-Job | Out-Null
    $workersThisRound.Job | Remove-Job

    $mergeArguments = @(
        "--bucket", "power",
        "--load-atlas", "cpp_mccfr\power64_v1.bin",
        "--start-street", "5",
        "--algorithm", "mccfr",
        "--ante", "$Ante",
        "--load", "$OutputModel",
        "--hands", "2",
        "--iterations", "0",
        "--seed", "$($Seed + $round)",
        "--save", "$next"
    )
    foreach ($item in $workersThisRound) {
        $mergeArguments += @("--merge", $item.Model)
    }
    & $exe @mergeArguments
    if ($LASTEXITCODE -ne 0) {
        throw "merge failed in parallel training round $round"
    }

    Move-Item -LiteralPath $next -Destination $OutputModel -Force
    $roundRoots = [uint64](($workersThisRound | Measure-Object -Property Roots -Sum).Sum)
    $completed += $roundRoots
    Set-Content -LiteralPath $progress -Value $completed
    foreach ($item in $workersThisRound) {
        Remove-Item -LiteralPath $item.Model, $item.Stdout, $item.Stderr -ErrorAction SilentlyContinue
    }
    $seconds = ((Get-Date) - $roundStarted).TotalSeconds
    Write-Host (
        "completed roots $completed/$TargetRoots; round roots=$roundRoots; " +
        "elapsed=$([Math]::Round($seconds, 1))s; " +
        "roots/s=$([Math]::Round($roundRoots / [Math]::Max(0.001, $seconds), 1))")
}
