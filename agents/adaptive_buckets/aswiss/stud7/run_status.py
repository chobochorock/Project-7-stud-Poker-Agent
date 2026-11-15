"""Progress of the arms of run_100m.ps1 / run_100m.sh (their -Action status / ACTION=status).

Usage: python run_status.py OUT --runs NAME:LABEL ... [--budget SECONDS] [--watch SECONDS] [--last 500]
Per arm: whether its process runs (OUT/NAME.pid), training roots, training CPU (and the share of --budget), tables in
use, resident memory (curve.csv, written every -CurveEvery roots); the online LBR of the average policy, mean of the
last --last deals with a 90% interval (online_lbr.csv, runs trained with -OnlineLbrEvery); evaluations made
(eval.jsonl, eval_final.jsonl) with the newest one; and for the two-layer accumulated-regret arms (ppfast_rt, ppfast_hid*) the
posterior waiting condition (pp_post.csv, one row every 10,000 roots: the window regret against the gate, judgments,
the root of the last one, tables, prior leaves, and whether the prior stopped at the memory budget).
--watch N repeats every N seconds (Ctrl+C ends it). Only reads files: the training is not touched.
LBR is a lower bound on exploitability, not exploitability.
"""
import argparse, csv, io, json, math, os, subprocess, sys, time

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("out")
ap.add_argument("--runs", nargs="+", required=True, help="NAME:LABEL")
ap.add_argument("--budget", type=float, default=0)
ap.add_argument("--watch", type=float, default=0)
ap.add_argument("--last", type=int, default=500)
a = ap.parse_args()


def alive(pid):
    if os.name == "nt":  # os.kill(pid, 0) would end the process on Windows
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"], capture_output=True, text=True).stdout
        return f'"{pid}"' in out
    try: os.kill(pid, 0); return True
    except OSError: return False


def tail(path, n):
    """the header and the last n complete lines of a CSV that may be growing"""
    if not os.path.exists(path): return None, []
    with open(path, "rb") as f:
        header = f.readline().decode().strip()
        f.seek(0, 2); end = f.tell(); size = 1 << 16; data = b""
        while end > 0 and data.count(b"\n") <= n + 1:
            step = min(size, end); end -= step; f.seek(end); data = f.read(step) + data; size *= 2
    lines = data.decode(errors="replace").split("\n")
    if not data.endswith(b"\n"): lines = lines[:-1]  # a partial last line is being written
    lines = [l.rstrip("\r") for l in lines]  # files written on Windows end their lines with CRLF
    lines = [l for l in lines if l and l != header][-n:]
    return header, lines


def rows(path, n):
    header, lines = tail(path, n)
    if not header: return []
    return [r for r in csv.DictReader(io.StringIO(header + "\n" + "\n".join(lines)))
            if None not in r.values() and not any(k == v for k, v in r.items())]  # never a header line as a row


def fnum(x, d=0): return f"{float(x):,.{d}f}"


def show():
    for spec in a.runs:
        name, label = spec.split(":", 1)
        d = os.path.join(a.out, name)
        pidf = os.path.join(a.out, name + ".pid")
        run = "running" if os.path.exists(pidf) and open(pidf).read().strip().isdigit() and alive(int(open(pidf).read())) else "not running"
        print(f"{name} ({label}): {run}")
        if not os.path.isdir(d): print("    not started"); continue
        c = rows(os.path.join(d, "curve.csv"), 1)
        if not c: print("    no curve row yet (curve.csv gets one every -CurveEvery roots)")
        if c:
            r = c[-1]
            cpu = r.get("train_cpu_seconds") or r.get("cpu_seconds") or r.get("seconds")
            share = f" ({100 * float(cpu) / a.budget:.1f}% of {a.budget:,.0f})" if a.budget > 0 and cpu else ""
            print(f"    {int(float(r['roots'])):,} roots, training CPU {fnum(cpu)} s{share}, {int(float(r['buckets'])):,} tables, "
                  f"{fnum(r.get('rss_mb', 0))} MB resident")
        snaps = rows(os.path.join(d, "snapshots.csv"), 10 ** 6)
        b = [r for r in snaps if r.get("kind") == "budget"]
        if b: print(f"    finished: CPU budget reached at {int(b[-1]['roots']):,} roots (budget snapshot {b[-1]['file']})")
        o = [r for r in rows(os.path.join(d, "online_lbr.csv"), 4 * a.last) if r["policy"] == "average"][-a.last:]
        if o:
            x = [float(r["lbr_pair"]) for r in o]
            m = sum(x) / len(x); sd = math.sqrt(sum((v - m) ** 2 for v in x) / max(1, len(x) - 1))
            print(f"    online LBR (average policy, last {len(x)} deals to {int(float(o[-1]['roots'])):,} roots): "
                  f"{m:.2f} +- {1.645 * sd / math.sqrt(len(x)):.2f} (90%)")
        for f, what in (("eval.jsonl", "snapshot evaluations"), ("eval_final.jsonl", "budget-snapshot evaluations")):
            p = os.path.join(d, f)
            if not os.path.exists(p): continue
            ev = [json.loads(l) for l in open(p) if l.strip()]
            if not ev: continue
            last = max(ev, key=lambda e: (int(e.get("roots", 0)), e.get("policy") == "average"))
            print(f"    {len(ev)} {what}; newest: {int(last.get('roots', 0)):,} roots, {last['policy']} {last['lbr_ante']:.3f} "
                  f"+- {last['lbr_se']:.3f} ({last.get('pairs', '?')} pairs)")
        pp = os.path.join(d, "pp_post.csv")
        if os.path.exists(pp):
            allr = rows(pp, 10 ** 9)
            if allr:
                p = allr[-1]; j = [r for r in allr if r["judged"] == "1"]
                lastj = f", last at {int(j[-1]['roots']):,} roots" if j else ""
                frozen = ", prior stopped at the memory budget" if p.get("prior_frozen") == "1" else ""
                noise = f" (beyond noise {float(p['window_regret_beyond_noise']):.3f}, the condition)" if p.get("window_regret_beyond_noise") else ""
                print(f"    posterior: window regret {float(p['window_regret']):.3f}{noise} ante (gate {p['gate']}) over "
                      f"{int(p['window_roots']):,} roots; {len(j)} judgments{lastj}; {int(p['buckets_after']):,} tables, "
                      f"{int(p['live_leaves']):,} prior leaves{frozen}")


while True:
    if a.watch > 0:
        if os.name == "nt": os.system("cls")
        else: print("\033[2J\033[H", end="")
        print(time.strftime("%Y-%m-%d %H:%M:%S"), f"(every {a.watch:g} s; Ctrl+C ends it)")
    show()
    sys.stdout.flush()
    if a.watch <= 0: break
    try: time.sleep(a.watch)
    except KeyboardInterrupt: break
