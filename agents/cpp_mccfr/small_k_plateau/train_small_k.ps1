# Usage (PowerShell, any cwd):
#   .\train_small_k.ps1 -K 16 -OutDir agents\cpp_mccfr\data\small_k_plateau_20261010\k16 [-TargetM 10] [-SeedOffset 0]
# Input:  bin\stud_mccfr_memory16.exe (the 2026-08-03 binary that trained the K512 reference).
# Output: OutDir\power<K>_epsheur20_memory16_v1.bin (atlas) and OutDir\k<K>_<budget>.bin checkpoints
#         (1k 3k 10k 30k 100k 300k 1m 2m .. <TargetM>m) plus one stdout/stderr log per process.
# What it does: fits a K-cluster power atlas with the K512 reference fit flags and seed (only --clusters
#   differs), then trains power-memory16 external-sampling MCCFR with the reference cut points
#   (10k -> 100k -> 1m -> 1M-root chunks), seeds and regret-prior schedule (made-call prior up to 1M roots,
#   none afterwards). With -K 512 -SeedOffset 0 the 10k/100k/1m/10m files should equal the reference files.
# Side checkpoints 1k, 3k (cold starts) and 30k, 300k (branches) are NOT on the main chain; seeds are ours.
# -SeedOffset shifts every training seed (replicate runs); the atlas fit seed stays 44101.
# Budget unit: 1 root iteration = 1 fresh 5th-street deal = 2 traversals. Node visits are in the stderr logs.
# -Divisor is for smoke tests only: it divides every root count.
param(
    [Parameter(Mandatory = $true)][int]$K,
    [Parameter(Mandatory = $true)][string]$OutDir,
    [int]$TargetM = 10,
    [uint64]$SeedOffset = 0,
    [int]$Divisor = 1
)
$ErrorActionPreference = "Stop"
$repo = "D:\Experiment\Project-7-stud-Poker-Agent"
Set-Location $repo
$exe = Join-Path $repo "agents\cpp_mccfr\bin\stud_mccfr_memory16.exe"
New-Item -ItemType Directory -Force $OutDir | Out-Null
$OutDir = (Resolve-Path -LiteralPath $OutDir).Path
$atlas = Join-Path $OutDir "power${K}_epsheur20_memory16_v1.bin"

function Invoke-Stud([string]$tag, [string[]]$arguments) {
    $o = Join-Path $OutDir "$tag.stdout.json"
    $e = Join-Path $OutDir "$tag.stderr.log"
    $p = Start-Process -FilePath $exe -ArgumentList $arguments -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $o -RedirectStandardError $e
    if ($p.ExitCode -ne 0) { throw "$tag failed with exit code $($p.ExitCode)" }
}

if (!(Test-Path -LiteralPath $atlas)) {
    Invoke-Stud "fit" @(
        '--bucket', 'power-memory16', '--fit-hands', '100000', '--clusters', "$K",
        '--power-samples', '128', '--fit-sample-cap', '100000', '--fit-policy-epsilon', '0.2',
        '--save-atlas', $atlas, '--start-street', '5', '--algorithm', 'mccfr', '--ante', '1000',
        '--hands', '2', '--iterations', '0', '--seed', '44101')
}

$common = @('--bucket', 'power-memory16', '--load-atlas', $atlas, '--start-street', '5',
    '--algorithm', 'mccfr', '--ante', '1000', '--hands', '2', '--iterations', '0')
$prior = @('--initial-regret-policy', 'made-call', '--initial-fold-regret', '1000',
    '--initial-other-regret', '1', '--made-min-category', '1')

function Train([string]$name, [string]$from, [uint64]$roots, [uint64]$report, [uint64]$seed, [bool]$withPrior, [bool]$chunk) {
    $model = Join-Path $OutDir "k${K}_$name.bin"
    if (Test-Path -LiteralPath $model) { return }
    $next = "$model.next"
    $a = $common + @('--root-iterations', "$roots", '--root-report-every', "$report")
    if ($chunk) { $a += @('--opponent', 'heuristic') }
    $a += @('--seed', "$($seed + $SeedOffset)", '--save', $next)
    if ($from) { $a += @('--load', (Join-Path $OutDir "k${K}_$from.bin")) }
    if ($withPrior) { $a += $prior }
    Invoke-Stud $name $a
    Move-Item -LiteralPath $next -Destination $model
}

# Main chain: cut points, seeds and prior flags of the K512 reference.
Train '10k'  ''     (10000 / $Divisor)  (5000 / $Divisor)   44201 $true $false
Train '100k' '10k'  (90000 / $Divisor)  (30000 / $Divisor)  44202 $true $false
# Side checkpoints (not in the reference; they do not feed the main chain).
Train '1k'   ''     (1000 / $Divisor)   (1000 / $Divisor)   44221 $true $false
Train '3k'   ''     (3000 / $Divisor)   (3000 / $Divisor)   44222 $true $false
Train '30k'  '10k'  (20000 / $Divisor)  (10000 / $Divisor)  44211 $true $false
Train '300k' '100k' (200000 / $Divisor) (100000 / $Divisor) 44212 $true $false
Train '1m'   '100k' (900000 / $Divisor) (100000 / $Divisor) 44203 $true $false
# 1M-root chunks without prior flags: seeds 44702..44710 to 10M, then 45010..45029 to 30M.
for ($c = 1; $c -lt $TargetM; ++$c) {
    $seed = if ($c -le 9) { 44701 + $c } else { 45000 + $c }
    Train "$($c + 1)m" "$($c)m" (1000000 / $Divisor) (1000000 / $Divisor) $seed $false $true
}
"DONE K=$K offset=$SeedOffset target=${TargetM}m $(Get-Date -Format s)" | Set-Content -LiteralPath (Join-Path $OutDir "TRAIN_DONE.txt")
