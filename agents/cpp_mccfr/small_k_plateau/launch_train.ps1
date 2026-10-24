# Usage (PowerShell):
#   .\launch_train.ps1 -Arms "1:0","2:0","16:1000" [-TargetM 10] [-Run agents\cpp_mccfr\data\small_k_plateau_20261010]
# Starts one single-threaded train_small_k.ps1 per "K:SeedOffset" arm in parallel and waits for all.
# Arm folder: <Run>\k<K> for offset 0, <Run>\k<K>_s<offset> otherwise. Prints one line per finished arm.
param(
    [Parameter(Mandatory = $true)][string[]]$Arms,
    [int]$TargetM = 10,
    [string]$Run = "D:\Experiment\Project-7-stud-Poker-Agent\agents\cpp_mccfr\data\small_k_plateau_20261010"
)
$driver = Join-Path $PSScriptRoot "train_small_k.ps1"
New-Item -ItemType Directory -Force $Run | Out-Null
$procs = @()
foreach ($arm in $Arms) {
    $k, $offset = $arm.Split(":")
    $name = if ([uint64]$offset -eq 0) { "k$k" } else { "k${k}_s$offset" }
    $out = Join-Path $Run $name
    New-Item -ItemType Directory -Force $out | Out-Null
    $p = Start-Process powershell -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $out "driver.stdout.log") -RedirectStandardError (Join-Path $out "driver.stderr.log") `
        -ArgumentList '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $driver, '-K', $k, '-OutDir', $out, '-TargetM', "$TargetM", '-SeedOffset', $offset
    "started $name pid=$($p.Id)"
    $procs += @{ name = $name; proc = $p }
}
foreach ($entry in $procs) {
    $entry.proc.WaitForExit()
    "finished $($entry.name) exit=$($entry.proc.ExitCode) $(Get-Date -Format s)"
}
