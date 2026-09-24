param(
    [string]$RunDir = "deep_cfr_compact\runs\reference_t100_k3000_v1",
    [int]$Port = 29741
)

$ErrorActionPreference = "Continue"
if (Test-Path Variable:PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}
$py = "C:\Users\choi\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
$log = Join-Path $RunDir "train.stdout.log"
$arguments = @(
  "-B", "deep_cfr_compact\train.py",
  "--run-dir", $RunDir,
  "--start-street", "5",
  "--iterations", "100",
  "--traversals", "3000",
  "--memory-capacity", "2000000",
  "--strategy-memory-capacity", "5000000",
  "--hidden", "256",
  "--layers", "2",
  "--batch-size", "2048",
  "--advantage-steps", "2000",
  "--policy-steps", "4000",
  "--advantage-scale", "32",
  "--save-policy-every", "5",
  "--generator-report-every", "100",
  "--threads", "1",
  "--port", "$Port",
  "--seed", "44001"
)
if (Test-Path (Join-Path $RunDir "checkpoint.pt")) {
    $arguments += "--resume"
}

& $py @arguments 2>&1 | Tee-Object -FilePath $log -Append

if ($LASTEXITCODE -ne 0) {
    throw "compact Deep CFR training failed with exit code $LASTEXITCODE"
}
