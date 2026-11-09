# Long 7-stud ASWISS comparison (default 100M roots), resumable. Run from the repository root:
#   powershell -ExecutionPolicy Bypass -File agents\adaptive_buckets\aswiss\stud7\run_100m.ps1                 # start or resume all arms
#   powershell -ExecutionPolicy Bypass -File agents\adaptive_buckets\aswiss\stud7\run_100m.ps1 -Action status   # progress
#   powershell -ExecutionPolicy Bypass -File agents\adaptive_buckets\aswiss\stud7\run_100m.ps1 -Action stop     # checkpoint and exit
#   powershell -ExecutionPolicy Bypass -File agents\adaptive_buckets\aswiss\stud7\run_100m.ps1 -Action eval     # LBR / gap of new snapshots
#   powershell -ExecutionPolicy Bypass -File agents\adaptive_buckets\aswiss\stud7\run_100m.ps1 -Action final    # newest snapshot, 20k pairs
#   powershell -ExecutionPolicy Bypass -File agents\adaptive_buckets\aswiss\stud7\run_100m.ps1 -Action presplit # epoch snapshots (taken right before each split), -PresplitParticles
#   powershell -ExecutionPolicy Bypass -File agents\adaptive_buckets\aswiss\stud7\run_100m.ps1 -Action plot     # figures
#   powershell -ExecutionPolicy Bypass -File agents\adaptive_buckets\aswiss\stud7\run_100m.ps1 -Action pack     # results -> results_data
#   powershell -ExecutionPolicy Bypass -File agents\adaptive_buckets\aswiss\stud7\run_100m.ps1 -Action build -ExeName aswiss_web.exe  # build only (web agent)
# Each arm is one process (one core). "run" again after a stop, crash or reboot resumes every unfinished arm from its
# checkpoint (bit-exact). -Threads K (K > 1) runs each arm's epoch-end average-policy label pass (--label avg arms:
# aswiss_b, aswiss_wb, hist_b, hist_wb, hist_bb, hist_bbf) on K threads; training stays on one. Any K > 1 gives the
# same results, but not those of -Threads 1 (other label deals), so an arm resumes only with the same choice.
# eval/final skip snapshots already evaluated, so they can be rerun while training continues.
# -OnlineLbrEvery N (N > 0): online test-then-train Policy-LBR every N roots (ReBeL: every -OnlineLbrRebelEvery hands)
# with -OnlineLbrParticles particles, into OUT/NAME/online_lbr.csv; its time is kept out of the training seconds.
# It is part of a run's settings: a run started without it cannot be resumed with it (use a new -Out).
# The first "run" builds aswiss_stud.exe with g++ found in this order: -Gxx PATH_TO_G++, g++ on PATH,
# C:\ProgramData\mingw64\mingw64\bin, C:\msys64\ucrt64\bin, C:\msys64\mingw64\bin, C:\mingw64\bin.
param(
    [ValidateSet("run", "build", "status", "stop", "eval", "final", "presplit", "plot", "pack")] [string]$Action = "run",
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
    [ValidateRange(1, 256)] [int]$Threads = 1,   # threads of the average-policy label pass (see above)
    [long]$OnlineLbrEvery = 0,       # online LBR every N roots (0: off; 1000 adds about 2% to a tabular arm)
    [long]$OnlineLbrRebelEvery = 10, # ReBeL: every N hands (one LBR pair costs about 1.5 training hands)
    [int]$OnlineLbrParticles = 16,
    [switch]$LogRegret,          # history/aswiss arms: regret.csv, the bucket regret total per root of each epoch window
    [switch]$LogRegretKmeans,    # the same for the k-means (atom) arms
    [switch]$LogSplits,          # history arms: splits.csv, one row per child bucket of every split (keys, label, z)
    [switch]$AverageOnly,        # eval/final: average policies only (MMD and ReBeL keep their one output policy)
    [int]$PresplitParticles = 256,
    [int]$PresplitPairs = 2000,
    # every arm given in -Arms: stop when the arm's training CPU time reaches this many seconds (budget snapshot and
    # checkpoint; equal-CPU-time comparison). Run again without it (or with a larger value) to continue towards -Roots.
    [double]$BudgetCpuSeconds = 0,
    # executable to build and use (relative to this folder): a second name lets new arms start with new code while
    # arms started earlier keep running with (and locking) aswiss_stud.exe
    [string]$ExeName = "aswiss_stud.exe",
    [long]$FreezeSplitsAt = 0,     # > 0: epochs ending at or after this root are not judged (no label pass, no split); tabular arms
    [string]$Gxx = ""   # path to g++.exe when it is not on PATH
)
$ErrorActionPreference = "Stop"
# powershell -File passes "-Arms aswiss,k256" as the single string "aswiss,k256", so split on commas here
$Arms = @($Arms | ForEach-Object { $_ -split "," } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Exe = Join-Path $Here $ExeName
$Atlas = Join-Path $Here "atlas"
$Arm = @{   # name, partition, atlas K, label, extra trainer arguments
    aswiss    = @("aswiss_k1024_z2", "aswiss", 1024, "ASWISS (context start, K=1024 atoms, z=2)", "")
    aswiss_b  = @("aswiss_b_k1024_z2", "aswiss", 1024, "ASWISS, method B labels (average policy, sampled)", "--label avg")
    aswiss_wb = @("aswiss_wb_k1024_z2", "aswiss", 1024, "ASWISS, window + method B labels", "--label window+avg")
    # history arms: label samples capped at 2.56M roots per epoch (history keys do not saturate; bounds the memory)
    hist      = @("history_street", "history", 1024, "bucket-history keys, window labels", "--history-cards street --label-cap 2560000")
    hist_b    = @("history_b_street", "history", 1024, "bucket-history keys, method B labels (average policy)", "--history-cards street --label avg --label-cap 2560000")
    hist_wb   = @("history_wb_street", "history", 1024, "bucket-history keys, window + method B labels", "--history-cards street --label window+avg --label-cap 2560000")
    hist_bb   = @("history_bb_street", "history", 1024, "ASWISS: bucket-history keys, method B, backward split", "--history-cards street --label avg --label-cap 2560000 --backward 1")
    hist_bbf  = @("history_bbf_street", "history", 1024, "ASWISS + hybrid average for labels (--avg-fill plain)", "--history-cards street --label avg --label-cap 2560000 --backward 1 --avg-fill plain")
    hist_every = @("history_every", "history", 1024, "bucket-history keys, cards at every decision", "--history-cards every --label-cap 2560000")
    k256      = @("atom_k256", "atom", 256, "k-means 256", "")
    k341      = @("atom_k341", "atom", 341, "k-means 341 (matched to ASWISS at 10M)", "")
    k512      = @("atom_k512", "atom", 512, "k-means 512", "")
    k1024     = @("atom_k1024", "atom", 1024, "k-means 1024 = atoms trained directly", "")
    # SOTA baseline: MMD self-play with a policy network (sota/mmd.hpp; the atlas K is unused)
    mmd       = @("net_mmd", "net", 1024, "MMD self-play (policy network)", "--algo mmd")
    escher    = @("net_escher", "net", 1024, "ESCHER (neural CFR)", "--algo escher")
    # ESCHER reservoir sensitivity: 1e6 samples per reservoir (default 1e5, the official code's) keeps every sample a
    # 24-CPU-hour run generates at the paper's Table 3 rate (~3,500 regret samples per iteration); ~1.7 GB when full
    escher_m1m = @("net_escher_m1m", "net", 1024, "ESCHER, reservoirs of 1e6", "--algo escher --escher-capacity 1000000")
    # ESCHER with its networks trained by gpu_trainer.py on the GPU (traversals and inference stay on this CPU core);
    # its budget is wall-clock (-BudgetCpuSeconds is passed as --budget-seconds: the training runs in a second process)
    escher_gpu = @("net_escher_gpu", "net", 1024, "ESCHER, networks trained on the GPU", "--algo escher")
    # scaled-down ReBeL (sota/rebel.hpp): roots = self-play hands, each solving up to three street subgames (~1 s);
    # its own budget and cadences override the 100M ones (60k hands ~ the wall-clock of a 100M tabular arm)
    rebel     = @("rebel", "rebel", 1024, "ReBeL (scaled: search + value networks)", "--roots 1000000 --epoch0 64 --checkpoint-every 500 --snapshot-every 0 --curve-every 100")
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

# build $Exe when it is missing or older than its sources (for example after git pull)
function BuildIfStale {
    $sources = @((Join-Path $Here "aswiss_stud.cpp"), (Join-Path $Here "h4_lbr_generated.hpp"),
                 (Join-Path $Here "..\..\..\cpp_mccfr\stud_mccfr.cpp")) +
               @(Get-ChildItem (Join-Path $Here "sota\*.hpp") | ForEach-Object { $_.FullName })
    $stale = -not (Test-Path $Exe)
    if (-not $stale) {
        $built = (Get-Item $Exe).LastWriteTime
        $stale = @($sources | Where-Object { (Get-Item $_).LastWriteTime -gt $built }).Count -gt 0
    }
    if ($stale) {
        $compiler = FindGxx
        # the compiler's own bin first, so a foreign libstdc++/libgcc DLL on PATH cannot shadow its tools
        $env:PATH = (Split-Path -Parent $compiler) + ";" + $env:PATH
        Write-Host "building $Exe with $compiler"
        # fully static (libstdc++, libgcc, winpthread): the program needs no MinGW DLL at run time
        & $compiler -O3 -std=c++17 -pthread -static (Join-Path $Here "aswiss_stud.cpp") -o $Exe
        if ($LASTEXITCODE -ne 0) {
            throw "build failed. If training processes are still running they lock $Exe : -Action stop, wait until -Action status shows running=False, then run again (the arms resume exactly)."
        }
    }
}

if ($Action -eq "build") {
    BuildIfStale
    Write-Host "$Exe is up to date"
}
elseif ($Action -eq "run") {
    BuildIfStale
    foreach ($a in $Arms) {
        $name = RunName $a; $dir = Join-Path $Out $name
        $pidFile = Join-Path $Out "$name.pid"
        if ((Test-Path $pidFile) -and (Get-Process -Id (Get-Content $pidFile) -ErrorAction SilentlyContinue)) {
            Write-Host "$name is already running (pid $(Get-Content $pidFile)); skipped"; continue
        }
        $argv = @("train", "--partition", $Arm[$a][1], "--atlas", (AtlasPath $Arm[$a][2]), "--z", "2", "--seed", $Seed,
                  "--roots", $Roots, "--epoch0", "10000", "--checkpoint-every", $CheckpointEvery,
                  "--snapshot-every", $SnapshotEvery, "--curve-every", $CurveEvery, "--h4", $H4, "--out", $dir)
        if ($Threads -gt 1) { $argv += @("--threads", $Threads) }
        if ($Arm[$a][4]) { $argv += $Arm[$a][4] -split " " }
        if (($LogRegret -and $Arm[$a][1] -in @("history", "aswiss")) -or ($LogRegretKmeans -and $Arm[$a][1] -eq "atom")) {
            $argv += @("--log-regret", "1")
        }
        if ($LogSplits -and $Arm[$a][1] -eq "history") { $argv += @("--log-splits", "1") }
        $gpu = $a -like "*_gpu"
        if ($BudgetCpuSeconds -gt 0) { $argv += @($(if ($gpu) { "--budget-seconds" } else { "--budget-cpu-seconds" }), $BudgetCpuSeconds) }
        if ($FreezeSplitsAt -gt 0 -and $Arm[$a][1] -in @("history", "aswiss")) { $argv += @("--freeze-splits-at", $FreezeSplitsAt) }
        if ($OnlineLbrEvery -gt 0) {
            $every = if ($Arm[$a][1] -eq "rebel") { $OnlineLbrRebelEvery } else { $OnlineLbrEvery }
            $argv += @("--online-lbr-every", $every, "--online-lbr-particles", $OnlineLbrParticles)
        }
        $log = Join-Path $Out "$name.$(Get-Date -Format yyyyMMdd-HHmmss).stderr.log"
        # Start-Process joins the arguments with spaces, so quote the ones that contain spaces (paths)
        $file = $Exe
        if ($gpu) {  # python gpu_trainer.py starts the trainer with its stdin/stdout as pipes
            $argv = @((Join-Path $Here "gpu_trainer.py"), "--backend", "torch", "--device", "cuda", "--log", (Join-Path $dir "gpu_log.csv"), "--", $Exe) + $argv
            $file = $Python
        }
        $quoted = $argv | ForEach-Object { if ("$_" -match '\s') { '"' + $_ + '"' } else { "$_" } }
        $p = Start-Process -FilePath $file -ArgumentList $quoted -RedirectStandardError $log -WindowStyle Hidden -PassThru
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
elseif ($Action -in @("eval", "final", "presplit")) {
    if (-not (Test-Path $Exe)) { throw "build first (-Action run builds $Exe)" }
    # @(...) keeps one arm an array: splatting a lone string (@specs) would pass it character by character
    $specs = @($Arms | ForEach-Object { "$(RunName $_):$($Arm[$_][1]):$(AtlasPath $Arm[$_][2])" })
    $extra = if ($Action -eq "final") { @("--latest", "--pairs", "20000", "--tag", "final") }
             elseif ($Action -eq "presplit") {  # epoch snapshots = the policy right before each split, many particles
                 @("--kinds", "epoch", "--min-roots", "0", "--particles", $PresplitParticles, "--pairs", $PresplitPairs,
                   "--rebel-pairs", $PresplitPairs, "--gap-roots", "0", "--only-average", "--tag", "presplit$PresplitParticles")
             } else { @("--min-roots", "100000") }
    if ($AverageOnly -and $Action -ne "presplit") { $extra += "--only-average" }
    # ReBeL searches at every street: its LBR hands run on $Workers threads inside one evaluation
    & $Python (Join-Path $Here "evaluate.py") $Exe $Out --runs @specs --workers $Workers --rebel-threads $Workers @extra
}
elseif ($Action -eq "plot") {
    $specs = @($Arms | ForEach-Object { "$(RunName $_):$($Arm[$_][3])" })
    # baseline: the first k-means arm that was run (k256 when present)
    $base = @("k256", "k341", "k512", "k1024") | Where-Object { $Arms -contains $_ } | Select-Object -First 1
    if (-not $base) { $base = "k256" }
    # the baseline is drawn even when -Arms leaves it out (e.g. -Arms hist_bbf on a folder that also holds k256)
    if ($Arms -notcontains $base -and (Test-Path (Join-Path $Out (RunName $base)))) { $specs += "$(RunName $base):$($Arm[$base][3])" }
    & $Python (Join-Path $Here "plot_stud.py") $Out (Join-Path $Out "results$Suffix") --runs @specs --baseline (RunName $base) --epoch-only
    # equal-CPU-time view: LBR against training CPU seconds (train_clock.py)
    & $Python (Join-Path $Here "plot_stud_time.py") $Out (Join-Path $Out "results${Suffix}_cputime.png") --runs @specs --clock cpu
    # pre-split evaluations (-Action presplit): the same figures from eval_presplit<P>.jsonl
    $pre = "eval_presplit$PresplitParticles.jsonl"
    if (@($Arms | Where-Object { Test-Path (Join-Path (Join-Path $Out (RunName $_)) $pre) }).Count -gt 0) {
        & $Python (Join-Path $Here "plot_stud.py") $Out (Join-Path $Out "results${Suffix}_presplit$PresplitParticles") --runs @specs --baseline (RunName $base) --epoch-only --eval-file $pre
        & $Python (Join-Path $Here "plot_stud_time.py") $Out (Join-Path $Out "results${Suffix}_presplit${PresplitParticles}_time.png") --runs @specs --eval-file $pre
        & $Python (Join-Path $Here "plot_stud_time.py") $Out (Join-Path $Out "results${Suffix}_presplit${PresplitParticles}_cputime.png") --runs @specs --eval-file $pre --clock cpu
    }
    # bucket regret (regret.csv) and memory (curve.csv)
    & $Python (Join-Path $Here "plot_regret_memory.py") $Out (Join-Path $Out "results${Suffix}_regret_memory.png") --runs @specs --baseline (RunName $base)
    # online LBR (runs trained with -OnlineLbrEvery): moving average of the last 100 and of the last 2000 deals
    $online = @($Arms | Where-Object { Test-Path (Join-Path (Join-Path $Out (RunName $_)) "online_lbr.csv") })
    if ($online.Count -gt 0) {
        $ospecs = @($online | ForEach-Object { "$(RunName $_):$($Arm[$_][3])" })
        $obase = if (Test-Path (Join-Path (Join-Path $Out (RunName $base)) "online_lbr.csv")) { @("--baseline", (RunName $base)) } else { @() }
        foreach ($w in @(100, 2000)) {
            & $Python (Join-Path $Here "plot_online_lbr.py") $Out (Join-Path $Out "results${Suffix}_online_w$w.png") --runs @ospecs --window $w @obase
            & $Python (Join-Path $Here "plot_online_lbr.py") $Out (Join-Path $Out "results${Suffix}_online_cpu_w$w.png") --runs @ospecs --window $w @obase --clock cpu
        }
    }
}
elseif ($Action -eq "pack") {
    # small result files of every run in $Out (no snapshots or checkpoints) -> results_data, for git or sharing
    # named after the output folder (runs_online -> results_data\runs_online_s1_summaries.tar.gz), so packing one
    # folder never overwrites another folder's archive
    $packName = Split-Path -Leaf (Resolve-Path $Out).Path
    & $Python (Join-Path $Here "pack_summaries.py") $Out (Join-Path $Here "results_data\${packName}${Suffix}_summaries.tar.gz")
}
