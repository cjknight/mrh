#!/usr/bin/env python3
"""Replay the gemm/gemm_batch calls recorded by LIBGPU's PROFILE_ML and rank them
by how much of the total gemm time each one accounts for.

Input lines look like (the name= field is verbatim -replay syntax):

  LIBGPU :: PROFILE_ML :: count= 3312  name= gemm N N 180 180 43200 180 43200 180 1 0
  LIBGPU :: PROFILE_ML :: count= 306   name= gemm_batch T N 16 16 180 180 180 16 1 0 240

Usage:
  ./replay_profile.py run.log                 # or: ... | ./replay_profile.py
  ./replay_profile.py run.log -n 20           # fewer timing iterations (faster)
  ./replay_profile.py run.log --exe ./a.out --threads 8
"""

import argparse
import re
import subprocess
import sys
from collections import OrderedDict

MARKER = "PROFILE_ML ::"
REC = re.compile(r"^\s*count=\s*(\d+)\s+name=\s*(\S.*?)\s*$")
TIME = re.compile(r"time=\s*([0-9.eE+-]+)\s*\[ms\]")
GFLOPS = re.compile(r"flops=\s*([0-9.eE+-]+)\s*\[GFlops/s\]")


def valid_replay(name):
    """True if name is a well-formed -replay argument string.

    Guards against three things seen in real logs: other LIBGPU printfs that also
    carry a count= field, MPI ranks interleaving mid-line, and truncated output.
    """
    f = name.split()
    if not f or f[0] not in ("gemm", "gemm_batch"):
        return False
    # gemm: 11 fields. gemm_batch: 12, or 15 once PROFILE_ML appends the strides.
    if f[0] == "gemm_batch":
        if len(f) not in (12, 15):
            return False
    elif len(f) != 11:
        return False
    if f[1] not in ("N", "T", "C") or f[2] not in ("N", "T", "C"):
        return False
    pos = f[3:9] + (f[11:12] if f[0] == "gemm_batch" else [])   # dims, lds, batch
    nonneg = f[12:15] if len(f) == 15 else []                    # strides (0 = broadcast)
    try:
        if any(int(v) <= 0 for v in pos) or any(int(v) < 0 for v in nonneg):
            return False
        float(f[9]), float(f[10])
    except ValueError:
        return False
    return True


def parse(stream):
    """-> (OrderedDict {replay_args: total_count}, {name: n_records}, [rejected]).

    Anchors on the PROFILE_ML marker and splits on it, so a line mangled by two
    ranks writing at once still yields the intact records it contains.
    """
    calls, seen, rejected, skipped = OrderedDict(), {}, [], 0
    for line in stream:
        if MARKER not in line:
            continue
        for frag in line.split(MARKER)[1:]:
            m = REC.match(frag)
            if not m:
                continue
            count, name = int(m.group(1)), m.group(2)
            if name.split()[:1] in (["gemv"], ["gemv_batch"]):
                skipped += 1        # recorded by PROFILE_ML but benchmark_gemm has no -replay for gemv
                continue
            if not valid_replay(name):
                rejected.append((count, name))
                continue
            calls[name] = calls.get(name, 0) + count
            seen[name] = seen.get(name, 0) + 1
    return calls, seen, rejected, skipped


def shape(name):
    """(mode, m, n, k, batch) from a replay arg string; batch=1 when not batched."""
    f = name.split()
    mode = f[0]
    m, n, k = int(f[3]), int(f[4]), int(f[5])
    batch = int(f[11]) if mode == "gemm_batch" and len(f) > 11 else 1
    return mode, m, n, k, batch


def bench(exe, name, num_iter, threads, timeout):
    """-> (per_call_ms, gflops) or (None, err) on failure."""
    cmd = [exe, "-replay"] + name.split() + ["-num_iter", str(num_iter)]
    env_prefix = {"OMP_NUM_THREADS": str(threads)} if threads else {}
    try:
        import os
        env = dict(os.environ, **env_prefix)
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return None, "timeout"
    if p.returncode != 0:
        tail = (p.stderr or p.stdout).strip().splitlines()
        return None, tail[-1][:60] if tail else f"exit {p.returncode}"
    t = TIME.search(p.stdout)
    g = GFLOPS.search(p.stdout)
    if not t:
        return None, "no time= line"
    return float(t.group(1)) / num_iter, (float(g.group(1)) if g else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?", help="file with PROFILE_ML lines (default: stdin)")
    ap.add_argument("-n", "--num-iter", type=int, default=100)
    ap.add_argument("--exe", default="./a.out")
    ap.add_argument("--threads", type=int, default=0, help="OMP_NUM_THREADS (0 = inherit)")
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()

    with open(args.log) if args.log else sys.stdin as fh:
        calls, seen, rejected, skipped = parse(fh)
    if not calls:
        sys.exit(f"no valid '{MARKER} count= ... name= ...' records found")

    if skipped:
        print(f"note: ignored {skipped} gemv/gemv_batch record(s) — not replayable", file=sys.stderr)
    for count, name in rejected:
        print(f"skipped malformed record: count={count} name={name!r}", file=sys.stderr)
    dup = {k: v for k, v in seen.items() if v > 1}
    if dup:
        r = max(dup.values())
        print(f"note: {len(dup)} shape(s) appeared in up to {r} records "
              f"(multiple MPI ranks or concatenated runs); counts are summed",
              file=sys.stderr)

    rows, failed = [], []
    for i, (name, count) in enumerate(calls.items(), 1):
        print(f"[{i}/{len(calls)}] {name}", file=sys.stderr, flush=True)
        per_call_ms, gflops = bench(args.exe, name, args.num_iter, args.threads, args.timeout)
        if per_call_ms is None:
            failed.append((name, count, gflops))
            continue
        mode, m, n, k, batch = shape(name)
        rows.append({
            "name": name, "count": count, "mode": mode,
            "mnk": f"{m}x{n}x{k}" + (f" x{batch}" if batch > 1 else ""),
            "per_call_ms": per_call_ms,
            "total_s": per_call_ms * count / 1000.0,
            "gflops": gflops,
            "gflop_call": 2.0 * m * n * k * batch / 1e9,
            "strided": len(name.split()) == 15,
        })

    if not rows:
        sys.exit("every replay failed; nothing to report")

    rows.sort(key=lambda r: -r["total_s"])
    grand = sum(r["total_s"] for r in rows)

    w = max(len(r["mnk"]) for r in rows)
    print(f"\n{'':2} {'mode':10} {'m x n x k [xbatch]':{w}}  {'count':>7} "
          f"{'per call':>10} {'total':>9} {'share':>7} {'cum':>7} {'GF/s':>7}")
    print("-" * (2 + 12 + w + 9 + 11 + 10 + 8 + 8 + 9))
    cum = 0.0
    for i, r in enumerate(rows, 1):
        frac = r["total_s"] / grand
        cum += frac
        print(f"{i:2} {r['mode']:10} {r['mnk']:{w}}  {r['count']:7} "
              f"{r['per_call_ms']:9.4f}m {r['total_s']:8.3f}s {frac:6.1%} {cum:6.1%} "
              f"{r['gflops']:7.1f}")
    print("-" * (2 + 12 + w + 9 + 11 + 10 + 8 + 8 + 9))
    tot_calls = sum(r["count"] for r in rows)
    print(f"{'':2} {'TOTAL':10} {'':{w}}  {tot_calls:7} {'':10} {grand:8.3f}s")

    # how few shapes cover most of the time
    cum = 0.0
    for i, r in enumerate(rows, 1):
        cum += r["total_s"] / grand
        if cum >= 0.9:
            print(f"\ntop {i} of {len(rows)} shapes = {cum:.1%} of gemm time")
            break

    guessed = [r for r in rows if r["mode"] == "gemm_batch" and not r["strided"]]
    if guessed:
        print(f"\nnote: {len(guessed)} batched shape(s) carried no strides in the log, so "
              f"the replay assumed packed batches.\n      Rebuild libgpu with -D_PROFILE_ML "
              f"from a current tree to record them; without them the timing for any\n"
              f"      broadcast operand (stride 0) is pessimistic.")

    if failed:
        print(f"\n{len(failed)} shape(s) did not run:")
        for name, count, why in failed:
            print(f"  count={count:<7} {name}   [{why}]")


if __name__ == "__main__":
    main()
