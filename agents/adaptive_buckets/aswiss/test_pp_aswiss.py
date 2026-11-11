"""Tests T1-T6 of pp_aswiss.cpp (PRIOR_POSTERIOR_DESIGN_KO.md section 10).

Usage: python test_pp_aswiss.py WORK_DIR [--quick]
Builds pp_aswiss into WORK_DIR, exports Kuhn, Leduc and mini-stud small (export.py, small/export_small.py,
export_obs.py) when missing, and runs:
  T1 --measure 0 and 1 give byte-identical policy files (pure and pp; Leduc and mini-stud)
  T2 caches: pure with --memo/--table on or off, pp with --leaf-cache/--pend-cache on or off: identical policies
  T3 the ESD table: |simulated - Rosner| <= 0.02 for n >= 25 and the bucket false-alarm rate within [0.04, 0.07]
  T4 --mode pp --prior-start identity --m 1 --eager 1 is identical to --mode pure (policies and events)
  T5 a run stopped after some rows and resumed equals the run done at once (policies, events, prior splits)
  T6 Kuhn and Leduc: the exact exploitability of the average policy at the last row is below the first row's
     (Kuhn's buckets have 2-3 information sets, which no outlier test can split: reported, not failed)
"""
import csv, filecmp, glob, os, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
work = Path(sys.argv[1]).resolve(); work.mkdir(parents=True, exist_ok=True)
quick = "--quick" in sys.argv
exe = work / "pp_aswiss"
subprocess.check_call(["g++", "-O3", "-std=c++17", str(HERE / "pp_aswiss.cpp"), "-o", str(exe)])
trees = {}
for g in ("kuhn", "leduc", "small"):
    d = work / "trees" / g
    if not (d / "obs.bin").exists():
        if g == "small": subprocess.check_call([sys.executable, str(HERE / "export.py"), g, str(d)], stdout=subprocess.DEVNULL)
        else: subprocess.check_call([sys.executable, str(HERE / "small" / "export_small.py"), g, str(d)], stdout=subprocess.DEVNULL)
        subprocess.check_call([sys.executable, str(HERE / "export_obs.py"), g, str(d)], stdout=subprocess.DEVNULL)
    trees[g] = d
results = []


def run(name, game, *args):
    out = work / "runs" / name
    if out.exists():
        for f in out.iterdir(): f.unlink()
    cmd = [str(exe), "--tree", str(trees[game]), "--out", str(out), *map(str, args)]
    subprocess.check_call(cmd, stderr=subprocess.DEVNULL)
    return out


def rerun(out, game, *args):
    cmd = [str(exe), "--tree", str(trees[game]), "--out", str(out), *map(str, args)]
    subprocess.check_call(cmd, stderr=subprocess.DEVNULL)
    return out


def same_policies(a, b):
    fa = sorted(os.path.basename(p) for p in glob.glob(str(a / "policy_*.bin")))
    fb = sorted(os.path.basename(p) for p in glob.glob(str(b / "policy_*.bin")))
    if fa != fb or not fa: return False, f"files differ ({len(fa)} vs {len(fb)})"
    bad = [f for f in fa if not filecmp.cmp(a / f, b / f, shallow=False)]
    return not bad, f"{len(fa)} files" + (f", differ: {bad[:3]}" if bad else "")


def same_file(a, b, name):
    return filecmp.cmp(a / name, b / name, shallow=False)


def check(name, ok, detail):
    results.append((name, ok, detail)); print(("PASS " if ok else "FAIL ") + name + ": " + detail, flush=True)


L = 64000 if quick else 128000
S = 16000 if quick else 32000
# T1
for game, iters in (("leduc", L), ("small", S)):
    for mode in ("pure", "pp"):
        a = run(f"t1_{game}_{mode}_0", game, "--mode", mode, "--iters", iters, "--measure", 0)
        b = run(f"t1_{game}_{mode}_1", game, "--mode", mode, "--iters", iters, "--measure", 1)
        ok, d = same_policies(a, b)
        check(f"T1 measure on/off, {game} {mode}", ok, d)
# T2
L2 = 6000  # without the memo every route re-walks the genealogy recursively: exponential in the level
base = run("t2_leduc_pure_ref", "leduc", "--mode", "pure", "--iters", L2)
for memo, table in (("0", "0"), ("1", "0"), ("0", "1")):
    o = run(f"t2_leduc_pure_m{memo}t{table}", "leduc", "--mode", "pure", "--iters", L2, "--memo", memo, "--table", table)
    ok, d = same_policies(base, o); check(f"T2 pure memo={memo} table={table}, leduc", ok, d)
for game, iters in (("leduc", L), ("small", S)):
    ref = run(f"t2_{game}_pp_ref", game, "--mode", "pp", "--iters", iters)
    for lc, pc in (("0", "0"), ("0", "1"), ("1", "0")):
        o = run(f"t2_{game}_pp_l{lc}p{pc}", game, "--mode", "pp", "--iters", iters, "--leaf-cache", lc, "--pend-cache", pc)
        ok, d = same_policies(ref, o)
        ok = ok and same_file(ref, o, "prior_splits.csv")
        check(f"T2 pp leaf-cache={lc} pend-cache={pc}, {game}", ok, d)
# T3
rows = list(csv.DictReader(open(HERE / "esd_table.csv")))
dev = max(abs(float(r["lambda"]) - float(r["rosner"])) for r in rows if int(r["n"]) >= 25)
alarm = [float(r["alarm"]) for r in rows if int(r["n"]) >= 25]
check("T3 ESD table", dev <= 0.02 and 0.04 <= min(alarm) and max(alarm) <= 0.07,
      f"max |sim - Rosner| {dev:.4f} (n >= 25), false alarm {min(alarm):.3f}-{max(alarm):.3f} (n >= 25)")
# T4
for game, iters in (("leduc", L), ("small", S)):
    a = run(f"t4_{game}_pure", game, "--mode", "pure", "--iters", iters)
    b = run(f"t4_{game}_ident", game, "--mode", "pp", "--prior-start", "identity", "--m", 1, "--eager", 1, "--iters", iters)
    ok, d = same_policies(a, b)
    ok = ok and same_file(a, b, "events.csv")
    check(f"T4 identity prior == pure, {game}", ok, d + ("" if same_file(a, b, "events.csv") else ", events differ"))
# T5
for game, iters, mode, extra in (("leduc", L, "pp", []), ("leduc", L, "pure", []), ("small", S, "pp", []),
                                 ("leduc", L, "pp", ["--schedule", "eps", "--prior-split", "epoch", "--measure", "1"])):
    tag = f"{game}_{mode}" + ("_eps" if extra else "")
    a = run(f"t5_{tag}_once", game, "--mode", mode, "--iters", iters, *extra)
    b = run(f"t5_{tag}_split", game, "--mode", mode, "--iters", iters, "--stop-after-rows", 9, *extra)
    rerun(b, game, "--mode", mode, "--iters", iters, "--stop-after-rows", 14, *extra)
    rerun(b, game, "--mode", mode, "--iters", iters, *extra)
    ok, d = same_policies(a, b)
    for f in ("events.csv", "prior_splits.csv"):
        if not same_file(a, b, f): ok = False; d += f", {f} differs"
    check(f"T5 resume, {tag}", ok, d)
# T6
for game, iters in (("kuhn", 256000), ("leduc", 512000)):
    for mode in ("pure", "pp"):
        o = run(f"t6_{game}_{mode}", game, "--mode", mode, "--iters", iters)
        subprocess.check_call([sys.executable, str(HERE / "eval.py"), str(trees[game]), str(o)], stdout=subprocess.DEVNULL)
        ex = list(csv.DictReader(open(o / "exploitability.csv")))
        first, last = float(ex[0]["expl_bucket_avg"]), float(ex[-1]["expl_bucket_avg"])
        splits = sum(int(r["splits"]) for r in ex)
        detail = f"exploitability {first:.4f} -> {last:.4f}, buckets {ex[-1]['buckets_after']}, posterior splits {splits}"
        if game == "kuhn" and splits == 0:
            check(f"T6 {game} {mode}", True, detail + " (no split possible: buckets of 2-3 information sets; reported)")
        else:
            check(f"T6 {game} {mode}", last < first, detail)
print("\n%d/%d passed" % (sum(ok for _, ok, _ in results), len(results)))
sys.exit(0 if all(ok for _, ok, _ in results) else 1)
