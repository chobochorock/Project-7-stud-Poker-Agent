# Long 7-stud ASWISS comparison (default 100M roots), resumable. Run from the repository root:
#   powershell -ExecutionPolicy Bypass -File agents\adaptive_buckets\aswiss\stud7\run_100m.ps1                 # start or resume all arms
#   powershell -ExecutionPolicy Bypass -File agents\adaptive_buckets\aswiss\stud7\run_100m.ps1 -Action status   # progress
#   powershell -ExecutionPolicy Bypass -File agents\adaptive_buckets\aswiss\stud7\run_100m.ps1 -Action stop     # checkpoint and exit
#   powershell -ExecutionPolicy Bypass -File agents\adaptive_buckets\aswiss\stud7\run_100m.ps1 -Action eval     # LBR / gap of new snapshots
#   powershell -ExecutionPolicy Bypass -File agents\adaptive_buckets\aswiss\stud7\run_100m.ps1 -Action final    # newest snapshot, 20k pairs
#   powershell -ExecutionPolicy Bypass -File agents\adaptive_buckets\aswiss\stud7\run_100m.ps1 -Action plot     # figures
# Each arm is one process (one core). "run" again after a stop, crash or reboot resumes every unfinished arm from its
# checkpoint (bit-exact). eval/final skip snapshots already evaluated, so they can be rerun while training continues.
# The first "run" builds aswiss_stud.exe with g++ found in this order: -Gxx PATH_TO_G++, g++ on PATH,
# C:\ProgramData\mingw64\mingw64\bin, C:\msys64\ucrt64\bin, C:\msys64\mingw64\bin, C:\mingw64\bin.
param(
    [ValidateSet("run", "status", "stop", "eval", "final", "plot")] [string]$Action = "run",
    [long]$Roots = 100000000,
    [int]$Seed = 1,
    [string[]]$Arms = @("aswiss", "k256", "k341", "k512", "k1024"),
    [string]$Out = "agents\adaptive_buckets\aswiss\stud7\data\runs_100m",
    [int]$Workers = 4,
    [string]$Python = "python",
    [ValidateSet("heuristic", "learn")] [string]$H4 = "heuristic",   # learn: 12-action 4th-street discard/reveal; names get _h4
    [long]$SnapshotEvery = 5000000,
    [long]$CheckpointEvery = 1000000,
    [long]$CurveEvery = 100000,
    [string]$Gxx = ""   # path to g++.exe when it is not on PATH
)
$ErrorActionPreference = "Stop"
# powershell -File passes "-Arms aswiss,k256" as the single string "aswiss,k256", so split on commas here
$Arms = @($Arms | ForEach-Object { $_ -split "," } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Exe = Join-Path $Here "aswiss_stud.exe"
$Atlas = Join-Path $Here "atlas"
$Arm = @{   # name, partition, atlas K, label
    aswiss = @("aswiss_k1024_z2", "aswiss", 1024, "ASWISS (context start, K=1024 atoms, z=2)")
    k256   = @("atom_k256", "atom", 256, "k-means 256")
    k341   = @("atom_k341", "atom", 341, "k-means 341 (matched to ASWISS at 10M)")
    k512   = @("atom_k512", "atom", 512, "k-means 512")
    k1024  = @("atom_k1024", "atom", 1024, "k-means 1024 = atoms trained directly")
}
$unknown = @($Arms | Where-Object { -not $Arm.ContainsKey($_) })
if ($unknown.Count -gt 0) { throw "unknown arm(s): $($unknown -join ', '); choose from: $($Arm.Keys -join ', ')" }
function AtlasPath($k) { Join-Path $Atlas "atlas_power${k}_heur_eps20_seed37301.bin" }
$Suffix = if ($H4 -eq "learn") { "_h4_s$Seed" } else { "_s$Seed" }
function RunName($a) { "$($Arm[$a][0])$Suffix" }
New-Item -ItemType Directory -Force $Out | Out-Null
$Out = (Resolve-Path $Out).Path   # absolute, so the training processes do not depend on their working directory

function FindGxx {
    if ($Gxx) { if (Test-Path $Gxx) { return (Resolve-Path $Gxx).Path } else { throw "-Gxx $Gxx does not exist" } }
    $onPath = Get-Command g++ -ErrorAction SilentlyContinue
    if ($onPath) { return $onPath.Source }
    foreach ($c in @("C:\ProgramData\mingw64\mingw64\bin\g++.exe", "C:\msys64\ucrt64\bin\g++.exe",
                     "C:\msys64\mingw64\bin\g++.exe", "C:\mingw64\bin\g++.exe")) {
        if (Test-Path $c) { return $c }
    }
    throw ("g++ not found. Pass -Gxx <path to g++.exe>, or install MinGW-w64 (for example MSYS2's " +
           "mingw-w64-ucrt-x86_64-gcc) and rerun.")
}

if ($Action -eq "run") {
    if (-not (Test-Path $Exe)) {
        $compiler = FindGxx
        # the compiler's own bin first, so a foreign libstdc++/libgcc DLL on PATH cannot shadow its tools
        $env:PATH = (Split-Path -Parent $compiler) + ";" + $env:PATH
        Write-Host "building $Exe with $compiler"
        # fully static (libstdc++, libgcc, winpthread): the program needs no MinGW DLL at run time
        & $compiler -O3 -std=c++17 -pthread -static (Join-Path $Here "aswiss_stud.cpp") -o $Exe
        if ($LASTEXITCODE -ne 0) { throw "build failed" }
    }
    foreach ($a in $Arms) {
        $name = RunName $a; $dir = Join-Path $Out $name
        $pidFile = Join-Path $Out "$name.pid"
        if ((Test-Path $pidFile) -and (Get-Process -Id (Get-Content $pidFile) -ErrorAction SilentlyContinue)) {
            Write-Host "$name is already running (pid $(Get-Content $pidFile)); skipped"; continue
        }
        $argv = @("train", "--partition", $Arm[$a][1], "--atlas", (AtlasPath $Arm[$a][2]), "--z", "2", "--seed", $Seed,
                  "--roots", $Roots, "--epoch0", "10000", "--checkpoint-every", $CheckpointEvery,
                  "--snapshot-every", $SnapshotEvery, "--curve-every", $CurveEvery, "--h4", $H4, "--out", $dir)
        $log = Join-Path $Out "$name.$(Get-Date -Format yyyyMMdd-HHmmss).stderr.log"
        # Start-Process joins the arguments with spaces, so quote the ones that contain spaces (paths)
        $quoted = $argv | ForEach-Object { if ("$_" -match '\s') { '"' + $_ + '"' } else { "$_" } }
        $p = Start-Process -FilePath $Exe -ArgumentList $quoted -RedirectStandardError $log -WindowStyle Hidden -PassThru
        Set-Content $pidFile $p.Id
        Write-Host "started $name (pid $($p.Id)) -> $dir"
    }
    Write-Host "progress: -Action status; stop safely: -Action stop"
}
elseif ($Action -eq "status") {
    foreach ($a in $Arms) {
        $name = RunName $a; $dir = Join-Path $Out $name
        $alive = $false
        $pidFile = Join-Path $Out "$name.pid"
        if (Test-Path $pidFile) { $alive = $null -ne (Get-Process -Id (Get-Content $pidFile) -ErrorAction SilentlyContinue) }
        $last = if (Test-Path "$dir\curve.csv") { (Get-Content "$dir\curve.csv" -Tail 1) } else { "-" }
        Write-Host ("{0,-22} running={1,-5} last curve row (roots,seconds,n,gap,se,buckets,rssMB): {2}" -f $name, $alive, $last)
    }
}
elseif ($Action -eq "stop") {
    foreach ($a in $Arms) { New-Item -ItemType File -Force (Join-Path (Join-Path $Out (RunName $a)) "STOP") | Out-Null }
    Write-Host "STOP files written; each arm saves a checkpoint within ~1000 roots and exits. Resume with -Action run."
}
elseif ($Action -in @("eval", "final")) {
    if (-not (Test-Path $Exe)) { throw "build first (-Action run builds $Exe)" }
    $specs = $Arms | ForEach-Object { "$(RunName $_):$($Arm[$_][1]):$(AtlasPath $Arm[$_][2])" }
    $extra = if ($Action -eq "final") { @("--latest", "--pairs", "20000", "--tag", "final") } else { @("--min-roots", "100000") }
    & $Python (Join-Path $Here "evaluate.py") $Exe $Out --runs @specs --workers $Workers @extra
}
elseif ($Action -eq "plot") {
    $specs = $Arms | ForEach-Object { "$(RunName $_):$($Arm[$_][3])" }
    & $Python (Join-Path $Here "plot_stud.py") $Out (Join-Path $Out "results$Suffix") --runs @specs --baseline (RunName "k256")
}
