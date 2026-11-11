# Prior-posterior ASWISS on mini-stud small, Windows PowerShell version of run_pp.sh (same arms, same outputs).
#   powershell -ExecutionPolicy Bypass -File run_pp.ps1 -Out D:\pp_runs -Seconds 1800 -Seeds 1,2,3 -Parallel 4
# Every arm trains single-threaded for -Seconds of CPU time. Arms (see run_pp.sh and PRIOR_POSTERIOR_DESIGN_KO.md):
#   ppeq ppeq_a pure pure_a pplen ppeq_fc mccfr, then kmeans / kmeans_a (k-means on ppeq's / ppeq_a's final buckets).
# Rerun the same command after an interruption: finished runs (exploitability.csv) are skipped and an unfinished one
# resumes from its state.bin. Needs Python with numpy and matplotlib, and g++ (MinGW-w64; -Gxx path when not on PATH).
param(
    [string]$Out = "pp_runs",
    [int]$Seconds = 1800,
    [int[]]$Seeds = @(1, 2, 3),
    [int]$Parallel = 4,
    [string]$Python = "python",
    [string]$Gxx = "",
    [string[]]$Arms = @("ppeq", "ppeq_a", "pure", "pure_a", "pplen", "ppeq_fc", "mccfr")
)
$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$Out = (Resolve-Path $Out).Path
$Tree = Join-Path $Out "tree"
$Exe = Join-Path $Out "pp_aswiss.exe"
$Esd = Join-Path $Here "esd_table.csv"

function FindGxx {
    if ($Gxx) { if (Test-Path $Gxx) { return (Resolve-Path $Gxx).Path } else { throw "-Gxx $Gxx does not exist" } }
    $onPath = Get-Command g++ -ErrorAction SilentlyContinue
    if ($onPath) { return $onPath.Source }
    foreach ($c in @("C:\ProgramData\mingw64\mingw64\bin\g++.exe", "C:\msys64\ucrt64\bin\g++.exe",
                     "C:\msys64\mingw64\bin\g++.exe", "C:\mingw64\bin\g++.exe")) { if (Test-Path $c) { return $c } }
    throw "g++ not found. Pass -Gxx <path to g++.exe>."
}
if (-not (Test-Path (Join-Path $Tree "obs.bin"))) {
    & $Python (Join-Path $Here "export.py") small $Tree | Out-Null
    & $Python (Join-Path $Here "export_obs.py") small $Tree
    if ($LASTEXITCODE -ne 0) { throw "export failed" }
}
$identity = Join-Path $Tree "bucket_identity.bin"
if (-not (Test-Path $identity)) {
    & $Python -c "import numpy as np,sys; n=len(np.fromfile(sys.argv[1]+'/counts.bin',np.int32)); np.arange(n,dtype=np.int32).tofile(sys.argv[1]+'/bucket_identity.bin')" $Tree
}
$src = Join-Path $Here "pp_aswiss.cpp"
if (-not (Test-Path $Exe) -or (Get-Item $src).LastWriteTime -gt (Get-Item $Exe).LastWriteTime) {
    $compiler = FindGxx
    $env:PATH = (Split-Path -Parent $compiler) + ";" + $env:PATH
    Write-Host "building $Exe with $compiler"
    & $compiler -O3 -std=c++17 -pthread -static $src -o $Exe
    if ($LASTEXITCODE -ne 0) { throw "build failed" }
}
$ArmArgs = @{
    "ppeq"    = @("--mode", "pp", "--prior-start", "equal", "--measure", "1")
    "ppeq_a"  = @("--mode", "pp", "--prior-start", "equal", "--outlier", "label", "--measure", "1")
    "pure"    = @("--mode", "pure", "--measure", "1")
    "pure_a"  = @("--mode", "pure", "--outlier", "label", "--measure", "1")
    "pplen"   = @("--mode", "pp", "--prior-start", "length", "--measure", "1")
    "ppeq_fc" = @("--mode", "pp", "--prior-start", "equal", "--scale", "fact-counterfact", "--measure", "1")
    "mccfr"   = @("--mode", "pure", "--p0", $identity, "--split", "0")
}
function RunJobs($jobs) {
    $running = @()
    foreach ($j in $jobs) {
        $d = Join-Path $Out "runs\$($j.Name)_s$($j.Seed)"
        if (Test-Path (Join-Path $d "exploitability.csv")) { continue }
        New-Item -ItemType Directory -Force -Path $d | Out-Null
        while (@($running | Where-Object { -not $_.HasExited }).Count -ge $Parallel) { Start-Sleep -Seconds 5 }
        $argv = @("--tree", $Tree, "--out", $d, "--seconds", "$Seconds", "--seed", "$($j.Seed)", "--esd-table", $Esd) + $j.Args
        $quoted = $argv | ForEach-Object { if ($_ -match "\s") { '"' + $_ + '"' } else { $_ } }
        Write-Host "start $($j.Name) seed $($j.Seed)"
        $running += Start-Process -FilePath $Exe -ArgumentList $quoted -RedirectStandardError (Join-Path $d "stderr.log") -WindowStyle Hidden -PassThru
    }
    while (@($running | Where-Object { -not $_.HasExited }).Count -gt 0) { Start-Sleep -Seconds 5 }
    foreach ($j in $jobs) {
        $d = Join-Path $Out "runs\$($j.Name)_s$($j.Seed)"
        if ((Test-Path (Join-Path $d "DONE")) -and -not (Test-Path (Join-Path $d "exploitability.csv"))) {
            & $Python (Join-Path $Here "eval.py") $Tree $d | Out-Null
        }
    }
}
$jobs = foreach ($s in $Seeds) { foreach ($a in $Arms) { [pscustomobject]@{ Name = $a; Seed = $s; Args = $ArmArgs[$a] } } }
RunJobs $jobs
$km = @()
foreach ($s in $Seeds) {
    foreach ($srcArm in @("ppeq", "ppeq_a")) {
        if ($Arms -notcontains $srcArm) { continue }
        $name = if ($srcArm -eq "ppeq") { "kmeans" } else { "kmeans_a" }
        $d = Join-Path $Out "runs\${name}_s$s"; New-Item -ItemType Directory -Force -Path $d | Out-Null
        $map = Join-Path $d "km_map.bin"
        if (-not (Test-Path $map)) {
            & $Python (Join-Path $Here "kmeans.py") $Tree (Join-Path $Out "runs\${srcArm}_s$s\final_bucket_map.bin") $map $s | Out-File (Join-Path $d "kmeans.log")
        }
        $km += [pscustomobject]@{ Name = $name; Seed = $s; Args = @("--mode", "pure", "--p0", $map, "--split", "0") }
    }
}
RunJobs $km
function ArmList($n) { ($Seeds | ForEach-Object { Join-Path $Out "runs\${n}_s$_" }) -join "," }
& $Python (Join-Path $Here "plot_pp.py") (Join-Path $Out "pp_small_esd") --arm "pp-equal=$(ArmList ppeq)" --arm "pp-length=$(ArmList pplen)" `
    --arm "pure=$(ArmList pure)" --arm "pp-equal fact-counterfact=$(ArmList ppeq_fc)" --ref "k-means=$(ArmList kmeans)" `
    --ref "MCCFR (no abstraction)=$(ArmList mccfr)" --kmeans k-means
& $Python (Join-Path $Here "plot_pp.py") (Join-Path $Out "pp_small_a") --arm "pp-equal, rule a=$(ArmList ppeq_a)" `
    --arm "pure, rule a=$(ArmList pure_a)" --ref "k-means (rule a counts)=$(ArmList kmeans_a)" `
    --ref "MCCFR (no abstraction)=$(ArmList mccfr)" --kmeans "k-means (rule a counts)"
Write-Host "done: $Out\pp_small_esd.md and $Out\pp_small_a.md"
