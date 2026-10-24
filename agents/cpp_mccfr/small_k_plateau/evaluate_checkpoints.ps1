# Usage (PowerShell):
#   .\evaluate_checkpoints.ps1 [-Parallel 6] [-Arms k1,k2] [-Budgets 3m,10m] [-Seeds 2026092401,2026092402] [-Wait] [-Pairs 2000] [-Particles 64]
# Input:  <Run>\<arm>\k<K>_<budget>.bin checkpoints and the arm's atlas (written by train_small_k.ps1).
# Output: <Run>\eval[_p<Particles>]\<arm>\<budget>\seed_<seed>\{local_gap.csv,lbr_pairs.csv,lbr_queries.csv} + eval.log
# Evaluator: agents\autoencoder_abstraction\bin\neural_mccfr.exe --evaluate-checkpoint (the binary behind the cached
#   K512 rows 1.593/1.114). Per run: root Local BR-gap (128 roots, 32 particles) and Policy-LBR on <Pairs> seat-swapped
#   deal pairs with <Particles> particles, current and average policy. Seeds 2026092401..05 are the K512-compatible ones
#   (the binary XORs 0x71834719 for LBR, giving deal streams 155250792, ...795, ...794, ...797, ...796).
# Metric: Local BR-gap in ante (plug-in, root decision only); LBR attacker profit in ante/hand (approximate lower bound
#   on exploitability, NOT exploitability). Higher = worse for the evaluated policy.
# Scheduling: jobs are ordered seed-first, so every checkpoint gets its first seed before any gets its second.
#   Finished outputs are skipped. With -Wait the script keeps polling for new checkpoints until every arm has
#   TRAIN_DONE.txt and all its jobs are done. Run from PowerShell only (the evaluator is dynamically linked).
param(
    [int]$Parallel = 6,
    [string[]]$Arms = @(),
    [string[]]$Budgets = @(),
    [uint64[]]$Seeds = @(2026092401, 2026092402, 2026092403, 2026092404, 2026092405),
    [int]$Pairs = 2000,
    [int]$Particles = 64,
    [switch]$Wait,
    [string]$Run = "D:\Experiment\Project-7-stud-Poker-Agent\agents\cpp_mccfr\data\small_k_plateau_20261010"
)
$ErrorActionPreference = "Stop"
Set-Location "D:\Experiment\Project-7-stud-Poker-Agent"
$exe = "D:\Experiment\Project-7-stud-Poker-Agent\agents\autoencoder_abstraction\bin\neural_mccfr.exe"
$evalRoot = if ($Particles -eq 64) { Join-Path $Run "eval" } else { Join-Path $Run "eval_p$Particles" }
$hands = @{ '1k' = 1000; '3k' = 3000; '10k' = 10000; '30k' = 30000; '100k' = 100000; '300k' = 300000 }

function Budget([string]$tag) {
    if ($hands.ContainsKey($tag)) { return $hands[$tag] }
    if ($tag -match '^(\d+)m$') { return [int64]$Matches[1] * 1000000 }
    throw "unknown budget tag $tag"
}

function IsDone([string]$out) {
    # lbr_pairs.csv is created when a run starts, so the log's final line is the completion record.
    $log = Join-Path (Split-Path $out -Parent) ((Split-Path $out -Leaf) + ".log")
    return (Test-Path -LiteralPath $log) -and [bool](Select-String -LiteralPath $log -Pattern 'DONE evaluation_only' -Quiet)
}

function IsFresh([string]$out) {
    # A log written in the last 3 minutes belongs to a run that is still going (possibly from another scheduler).
    $log = Join-Path (Split-Path $out -Parent) ((Split-Path $out -Leaf) + ".log")
    return (Test-Path -LiteralPath $log) -and ((Get-Date) - (Get-Item -LiteralPath $log).LastWriteTime).TotalSeconds -lt 180
}

function Pending {
    $jobs = @()
    $dirs = Get-ChildItem -LiteralPath $Run -Directory | Where-Object { $_.Name -match '^k\d+(_s\d+)?$' }
    if ($Arms.Count) { $dirs = $dirs | Where-Object { $Arms -contains $_.Name } }
    foreach ($dir in $dirs) {
        $atlas = Get-ChildItem -LiteralPath $dir.FullName -Filter 'power*_epsheur20_memory16_v1.bin' | Select-Object -First 1
        if (!$atlas) { continue }
        foreach ($model in Get-ChildItem -LiteralPath $dir.FullName -Filter 'k*_*.bin') {
            if ($model.Name -notmatch '^k\d+_(\w+)\.bin$') { continue }
            $tag = $Matches[1]
            if ($Budgets.Count -and ($Budgets -notcontains $tag)) { continue }
            foreach ($seed in $Seeds) {
                $out = Join-Path $evalRoot "$($dir.Name)\$tag\seed_$seed"
                if ((IsDone $out) -or (IsFresh $out)) { continue }
                if (Test-Path -LiteralPath (Join-Path (Split-Path $out -Parent) "seed_$seed.failed")) { continue }
                $jobs += [pscustomobject]@{ arm = $dir.Name; tag = $tag; seed = $seed; budget = (Budget $tag)
                    atlas = $atlas.FullName; model = $model.FullName; out = $out }
            }
        }
    }
    return $jobs | Sort-Object seed, budget, arm
}

function AllTrained {
    $dirs = Get-ChildItem -LiteralPath $Run -Directory | Where-Object { $_.Name -match '^k\d+(_s\d+)?$' }
    if ($Arms.Count) { $dirs = $dirs | Where-Object { $Arms -contains $_.Name } }
    foreach ($dir in $dirs) { if (!(Test-Path -LiteralPath (Join-Path $dir.FullName 'TRAIN_DONE.txt'))) { return $false } }
    return $true
}

$running = @()
$failed = 0
while ($true) {
    $still = @()
    foreach ($r in $running) {
        if (!$r.proc.HasExited) { $still += $r; continue }
        # The result file is the completion record; a run that exited without it is marked and not retried.
        if (!(IsDone $r.job.out)) {
            $failed++
            Set-Content -LiteralPath (Join-Path (Split-Path $r.job.out -Parent) "seed_$($r.job.seed).failed") -Value (Get-Date -Format s)
            "FAILED $($r.job.arm) $($r.job.tag) seed=$($r.job.seed)"
        }
    }
    $running = $still
    $active = @{}
    foreach ($r in $running) { $active[$r.job.out] = $true }
    $queue = @(Pending | Where-Object { !$active.ContainsKey($_.out) })
    foreach ($job in $queue) {
        if ($running.Count -ge $Parallel) { break }
        $parent = Split-Path $job.out -Parent
        New-Item -ItemType Directory -Force $parent | Out-Null
        if (Test-Path -LiteralPath $job.out) {
            # A stale directory without a DONE log is an interrupted run; the evaluator refuses an existing NEW_OUT.
            Rename-Item -LiteralPath $job.out -NewName ("seed_$($job.seed).interrupted_" + (Get-Date -Format HHmmss))
        }
        $log = Join-Path $parent "seed_$($job.seed).log"
        $arguments = @('--evaluate-checkpoint', $job.atlas, $job.model, $job.out, "$($job.budget)", 'power-memory16',
            '128', '32', "$Pairs", "$Particles", "$($job.seed)")
        $p = Start-Process -FilePath $exe -ArgumentList $arguments -NoNewWindow -PassThru `
            -RedirectStandardOutput $log -RedirectStandardError "$log.err"
        $running += [pscustomobject]@{ proc = $p; job = $job }
    }
    # Re-list after the training-done check so a checkpoint written in between is not missed.
    if ($running.Count -eq 0 -and $queue.Count -eq 0 -and (!$Wait -or (AllTrained)) -and @(Pending).Count -eq 0) { break }
    Start-Sleep -Seconds 3
}
"EVAL DONE failed=$failed $(Get-Date -Format s)"
