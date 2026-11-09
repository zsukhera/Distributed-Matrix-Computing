#!/usr/bin/env python3
"""summarize_results.py — turn results/q{1,2,3}_outputs.txt into results/summary.md.

Every timing configuration is run REPEATS times by run_qN.sh; this script keeps
the BEST run (smallest total time) of each configuration, checks every printed
value against the assignment tables, and writes Markdown tables:
    python3 summarize_results.py            -> results/summary.md
"""
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("RESULTS_DIR", os.path.join(HERE, "results"))

# ----------------------------------------------------------------------------
# expected values
TABLE1A = [
    [0, 6, -8, -2, -11, 0, 6, -8], [17, -11, 1, -2, -10, 17, -11, 1],
    [-8, 0, 3, -9, 19, -8, 0, 3], [-5, 4, -2, 12, 6, -5, 4, -2],
    [5, -6, -7, 12, -14, 5, -6, -7], [1, -9, 16, -9, 1, 1, -9, 16],
    [-10, 16, -3, -2, 9, -10, 16, -3], [0, 6, -8, -2, -11, 0, 6, -8]]
TABLE1B = {1000: [2, -2, 7, 3, 9], 2000: [5, -10, 15, -12, -5], 4000: [-7, 5, 20, 9, 17]}
TABLE4A = {512: [33, 24, 27], 1024: [33, 5, -20]}
TABLE4B = {512: ["0.0001220703125", "-0.000152587890625", "-0.00079345703125"],
           1024: ["0.0001220703125", "-0.000152587890625", "0.0008697509765625"]}


def blocks(path):
    """Yield (command, p, threads, args, text, repeat_no) for every run in a log."""
    if not os.path.exists(path):
        return []
    raw = open(path, errors="replace").read()
    stopped = "[STOPPED MANUALLY" in raw     # last run was killed: keep its partial output
    lines = raw.split("\n")
    out, i, repno = [], 0, 0
    while i < len(lines):
        m = re.match(r"######## REPEAT (\d+)/", lines[i])
        if m:
            repno = int(m.group(1))
        if lines[i].startswith("$ ") and i > 0 and lines[i - 1].startswith("====="):
            cmd = lines[i][2:]
            j = i + 2
            body = []
            while j < len(lines) and not lines[j].startswith("[exit="):
                body.append(lines[j])
                j += 1
            if j >= len(lines) and not stopped:   # run still in progress
                break
            ex = lines[j] if j < len(lines) else "[exit=killed]"
            p = re.search(r"-n (\d+)", cmd)
            t = re.search(r"OMP_NUM_THREADS=(\d+)", cmd)
            args = cmd.split("hosts ./")[-1].split()[1:] if "hosts ./" in cmd else []
            out.append(dict(cmd=cmd, p=int(p.group(1)) if p else 1,
                            t=int(t.group(1)) if t else 1, args=args,
                            text="\n".join(body), exit=ex, rep=repno))
            i = j
        i += 1
    return out


def fnum(rx, text, default=None):
    m = re.search(rx, text)
    return float(m.group(1)) if m else default


def per_rank(text):
    """Rows of a 'Per-process times' table -> list of (rank, a, b)."""
    rows = []
    if "Per-process times" not in text:
        return rows
    for ln in text.split("Per-process times", 1)[1].split("\n")[2:]:
        f = ln.split()
        if len(f) != 3 or not f[0].isdigit():
            break
        rows.append((int(f[0]), float(f[1]), float(f[2])))
    return rows


def fmt(x, nd=4):
    if x is None:
        return "–"
    if isinstance(x, str):
        return x
    if x == 0:
        return "0"
    if abs(x) < 10 ** -nd:
        return f"{x:.2e}"
    return f"{x:.{nd}f}"


def md_table(header, rows):
    s = "| " + " | ".join(header) + " |\n|" + "|".join("---" for _ in header) + "|\n"
    for r in rows:
        s += "| " + " | ".join(fmt(c) if not isinstance(c, str) else c for c in r) + " |\n"
    return s + "\n"


def best(runs, key="total"):
    runs = [r for r in runs if r.get(key) is not None]
    return min(runs, key=lambda r: r[key]) if runs else None


# ============================================================================
# Q1
def entries5(text, N):
    """Five Table 1(b) entries in order, whichever format was printed."""
    m = re.search(r"C\(0,0\)=(-?\d+)\s+C\(0,1\)=(-?\d+)\s+C\(1,0\)=(-?\d+)\s+"
                  r"C\(N/2,N/2\)=(-?\d+)\s+C\(N-1,N-1\)=(-?\d+)", text)
    if m:
        return [int(x) for x in m.groups()]
    want = [(0, 0), (0, 1), (1, 0), (N // 2, N // 2), (N - 1, N - 1)]
    vals = []
    for r, c in want:
        mm = re.search(rf"C\({r},{c}\)=(-?\d+)", text)
        vals.append(int(mm.group(1)) if mm else None)
    return vals


def matrices8(text):
    out = []
    for m in re.finditer(r"Full matrix C \(N=8\):\n((?:[ \t\-\d]+\n){8})", text + "\n"):
        out.append([[int(x) for x in ln.split()] for ln in m.group(1).strip().split("\n")])
    return out


def q1(md):
    bl = blocks(os.path.join(RES, "q1_outputs.txt"))
    if not bl:
        md.append("_results/q1_outputs.txt not found_\n")
        return {}
    data = defaultdict(list)
    corr8 = []
    for b in bl:
        if not b["args"]:
            continue
        N, task = int(b["args"][0]), b["args"][1]
        tx = b["text"]
        if N == 8:
            mats = matrices8(tx)
            ok = len(mats) > 0 and all(mm == TABLE1A for mm in mats)
            corr8.append((task, b["p"], b["t"], len(mats), "OK" if ok else "MISMATCH"))
            continue
        e = entries5(tx, N)
        r = dict(N=N, p=b["p"], t=b["t"], rep=b["rep"], entries=e,
                 ok=(e == TABLE1B.get(N)), ranks=per_rank(tx))
        if task == "1.1":
            r.update(ijk=fnum(r"i-j-k \(serial\):\s+([\d.e+-]+)", tx),
                     ikj=fnum(r"i-k-j \(serial\):\s+([\d.e+-]+)", tx),
                     omp=fnum(r"i-k-j \(OpenMP x\d+\):\s+([\d.e+-]+)", tx))
            r["total"] = r["omp"]
            ents = re.findall(r"C\(0,0\)=(-?\d+)\s+C\(0,1\)=(-?\d+)\s+C\(1,0\)=(-?\d+)\s+"
                              r"C\(N/2,N/2\)=(-?\d+)\s+C\(N-1,N-1\)=(-?\d+)", tx)
            r["ok"] = len(ents) == 3 and all([int(x) for x in g] == TABLE1B[N] for g in ents)
        elif task == "1.2":
            r.update(scatter=fnum(r"Scatterv\(A\):\s+([\d.e+-]+)", tx),
                     bcast=fnum(r"Bcast\(B\):\s+([\d.e+-]+)", tx),
                     mult=fnum(r"Multiply:\s+([\d.e+-]+)", tx),
                     gather=fnum(r"Gatherv\(C\):\s+([\d.e+-]+)", tx),
                     total=fnum(r"Total:\s+([\d.e+-]+)", tx),
                     memB=fnum(r"Memory for B per process:\s+([\d.e+-]+)", tx))
        elif task == "1.3":
            r.update(ring=fnum(r"Ring-comm:\s+([\d.e+-]+)", tx),
                     mult=fnum(r"Multiply:\s+([\d.e+-]+)", tx),
                     total=fnum(r"Total:\s+([\d.e+-]+)", tx),
                     mem=fnum(r"Memory/proc \(2 B-blocks\):\s+([\d.e+-]+)", tx))
        elif task == "1.4":
            r.update(mult=fnum(r"Multiply:\s+([\d.e+-]+)", tx),
                     wait=fnum(r"Wait\(comm\):\s+([\d.e+-]+)", tx),
                     total=fnum(r"Total:\s+([\d.e+-]+)", tx),
                     mem=fnum(r"Memory/proc \(2 B-blocks\):\s+([\d.e+-]+)", tx))
            steps = re.findall(r"^\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s*$",
                               tx.split("Per-step times")[-1].split("Per-process")[0], re.M)
            r["steps"] = [(int(a), float(b_), float(c)) for a, b_, c in steps]
        data[(task, N, b["p"], b["t"])].append(r)

    md.append("## Question 1 — Dense matrix multiplication\n")
    md.append("### Q1 correctness, N = 8 (whole matrix compared with Table 1(a))\n")
    md.append(md_table(["task", "processes", "threads", "matrices printed", "Table 1(a)"],
                       [(t, str(p), str(th), str(n), ok) for t, p, th, n, ok in corr8]))

    allok = all(r["ok"] for v in data.values() for r in v)
    nruns = sum(len(v) for v in data.values())
    md.append(f"Large-N runs parsed: {nruns}; every printed five-entry set matches Table 1(b): "
              f"**{'YES' if allok else 'NO'}**\n\n")

    # Task 1.1
    md.append("### Task 1.1 — loop order and OpenMP (best of repeats, seconds)\n")
    rows = []
    for (task, N, p, t), runs in sorted(data.items()):
        if task != "1.1":
            continue
        ijk = min(r["ijk"] for r in runs)
        ikj = min(r["ikj"] for r in runs)
        omp = min(r["omp"] for r in runs)
        rows.append((str(N), str(t), str(len(runs)), ijk, ikj, omp,
                     f"{ijk / ikj:.2f}x", f"{ikj / omp:.2f}", f"{ikj / omp / t:.2f}",
                     "OK" if all(r["ok"] for r in runs) else "MISMATCH"))
    md.append(md_table(["N", "threads", "runs", "i-j-k serial", "i-k-j serial", "i-k-j OpenMP",
                        "i-j-k / i-k-j", "OpenMP speedup vs i-k-j", "efficiency", "Table 1(b)"], rows))

    # Task 1.2
    md.append("### Task 1.2 — Version A step times (best run by total, seconds)\n")
    rows = []
    for (task, N, p, t), runs in sorted(data.items()):
        if task != "1.2":
            continue
        r = best(runs)
        rows.append((str(N), str(p), str(len(runs)), r["scatter"], r["bcast"], r["mult"],
                     r["gather"], r["total"], fmt(r["memB"], 1),
                     "OK" if all(x["ok"] for x in runs) else "MISMATCH"))
    md.append(md_table(["N", "p", "runs", "Scatterv(A)", "Bcast(B)", "Multiply", "Gatherv(C)",
                        "Total", "B per process (MiB)", "Table 1(b)"], rows))

    # Task 1.3
    md.append("### Task 1.3 — Version B blocking ring (best run, seconds)\n")
    rows = []
    for (task, N, p, t), runs in sorted(data.items()):
        if task != "1.3":
            continue
        r = best(runs)
        a = best(data.get(("1.2", N, p, t), []))
        rows.append((str(N), str(p), str(len(runs)), r["mult"], r["ring"], r["total"],
                     a["total"] if a else None, fmt(r["mem"], 2), fmt(a["memB"], 1) if a else "–",
                     "OK" if all(x["ok"] for x in runs) else "MISMATCH"))
    md.append(md_table(["N", "p", "runs", "Multiply (max)", "Ring comm (max)", "Total B",
                        "Total A (same p)", "B-buffers/proc MiB (B)", "full B MiB (A)", "Table 1(b)"], rows))

    # Task 1.4
    md.append("### Task 1.4 — Version C non-blocking ring (best run, seconds)\n")
    rows = []
    for (task, N, p, t), runs in sorted(data.items()):
        if task != "1.4":
            continue
        r = best(runs)
        b_ = best(data.get(("1.3", N, p, 1), []))
        rows.append((str(N), f"{p} x {t}", str(len(runs)), r["mult"], r["wait"], r["total"],
                     b_["ring"] if b_ else None, b_["total"] if b_ else None,
                     "OK" if all(x["ok"] for x in runs) else "MISMATCH"))
    md.append(md_table(["N", "p x threads", "runs", "Multiply (max)", "Wait (max)", "Total C",
                        "Version B ring comm", "Version B total", "Table 1(b)"], rows))

    # Speedup / efficiency vs serial i-k-j (N=2000, 4000)
    md.append("### Q1 speedup and efficiency (vs serial i-k-j of Task 1.1, best runs)\n")
    rows = []
    for N in (2000, 4000):
        ser = data.get(("1.1", N, 1, 1))
        if not ser:
            continue
        ts = min(r["ikj"] for r in ser)
        rows.append((str(N), "serial i-k-j", "1 x 1", ts, "1.00", "1.00"))
        for (task, NN, p, t), runs in sorted(data.items()):
            if NN != N or task == "1.1":
                continue
            r = best(runs)
            name = {"1.2": "Version A", "1.3": "Version B", "1.4": "Version C"}[task]
            if task == "1.4" and t > 1:
                name += " hybrid"
            cores = min(p, 3)
            rows.append((str(N), name, f"{p} x {t}", r["total"], f"{ts / r['total']:.2f}",
                         f"{ts / r['total'] / cores:.2f}"))
    md.append("Efficiency = speedup / physical cores used (min(p,3); the cluster has 3 cores).\n\n")
    md.append(md_table(["N", "version", "p x threads", "time (s)", "speedup", "efficiency"], rows))

    # Table 2 layout for N=4000
    md.append("### Table 2 — all versions side by side, N = 4000\n")
    rows = []
    E = TABLE1B[4000]
    s1 = data.get(("1.1", 4000, 1, 1))
    s4 = data.get(("1.1", 4000, 1, 4))
    if s1:
        rows.append(("serial (i-k-j)", "1 x 1", *[str(v) for v in s1[0]["entries"]], min(r["ikj"] for r in s1)))
    if s4:
        rows.append(("OpenMP", "1 x 4", *[str(v) for v in s4[0]["entries"]], min(r["omp"] for r in s4)))
    for task, name in (("1.2", "Version A"), ("1.3", "Version B"), ("1.4", "Version C")):
        for (tk, N, p, t), runs in sorted(data.items()):
            if tk == task and N == 4000 and (t == 1 and p in (3, 8) or t > 1):
                r = best(runs)
                nm = name + (", hybrid" if t > 1 or (task == "1.4" and p == 3) else "")
                rows.append((nm, f"{p} x {t}", *[str(v) for v in r["entries"]], r["total"]))
    md.append(f"Expected (Table 1(b), N=4000): {E}\n\n")
    md.append(md_table(["Version", "processes x threads", "C(0,0)", "C(0,1)", "C(1,0)",
                        "C(N/2,N/2)", "C(N-1,N-1)", "time (s)"], rows))

    # per-rank data for bar charts
    md.append("### Q1 per-process computation / communication (best run, N = 4000; seconds)\n")
    md.append("Version A: comm = Scatterv + Bcast + Gatherv. Version B: comm = blocking "
              "Send/Recv. Version C: comm = time in MPI_Waitall.\n\n")
    rows = []
    for task, name in (("1.2", "A"), ("1.3", "B"), ("1.4", "C")):
        for (tk, N, p, t), runs in sorted(data.items()):
            if tk == task and N == 4000 and p in (4, 8, 3, 2):
                r = best(runs)
                for rk, c, m in r["ranks"]:
                    rows.append((name, f"{p} x {t}", str(rk), c, m))
    md.append(md_table(["version", "p x threads", "rank", "compute (s)", "comm (s)"], rows))

    md.append("### Version C per-step multiply / wait (max over ranks), N = 4000, p = 8, best run\n")
    r = best(data.get(("1.4", 4000, 8, 1), []))
    if r:
        md.append(md_table(["step", "multiply (s)", "wait (s)"],
                           [(str(s), a, b_) for s, a, b_ in r["steps"]]))
    return data


# ============================================================================
# Q2
def q2_sections(text):
    """Split an MPI run into '--- Title: n=.., p=.., threads=.. ---' sections."""
    secs = []
    parts = re.split(r"\n(?=--- )", "\n" + text)
    for part in parts:
        m = re.match(r"--- (Task 2\.[234][^:]*): n=(\d+), p=(\d+), threads=(\d+) ---", part)
        if not m:
            continue
        s = dict(title=m.group(1), n=int(m.group(2)), p=int(m.group(3)), t=int(m.group(4)))
        rows = re.findall(r"^\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+) KiB",
                          part, re.M)
        s["ranks"] = [dict(rank=int(a), rows=int(b), links=int(c), compute=float(d), comm=float(e),
                           wait=float(f), kib=float(g)) for a, b, c, d, e, f, g in rows]
        s["total"] = fnum(r"Total time:\s+([\d.]+)", part)
        s["step"] = fnum(r"Time per step:\s+([\d.]+)", part)
        s["imb"] = fnum(r"Compute imbalance \(max/avg\):\s+([\d.]+)", part)
        s["links_ok"] = "Total links" in part and "MISMATCH" not in part.split("Compute imbalance")[0]
        s["H"] = re.findall(r"H\((\d+),(\d+)\) = ([\d.]+)\s+\(Table 3: [\d.]+\s+(OK|MISMATCH)\)", part)
        s["ok"] = len(s["H"]) == 3 and all(h[3] == "OK" for h in s["H"]) and s["links_ok"]
        secs.append(s)
    return secs


def q2(md):
    bl = blocks(os.path.join(RES, "q2_outputs.txt"))
    if not bl:
        md.append("_results/q2_outputs.txt not found_\n")
        return {}
    t21 = defaultdict(list)
    mpi = defaultdict(list)
    bounds = {}
    corr = []
    for b in bl:
        if not b["args"]:
            continue
        n, task = int(b["args"][0]), b["args"][1]
        tx = b["text"]
        for blk in re.split(r"\n(?=--- Task 2\.1: )", tx):
            m = re.search(r"--- Task 2\.1: n=(\d+), threads=(\d+) ---", blk)
            if not m:
                continue
            r = dict(n=int(m.group(1)), t=int(m.group(2)), rep=b["rep"])
            for key, lab in (("serial", r"Serial \(Code 3\)"), ("static", r"OpenMP schedule\(static\)"),
                             ("dynamic", r"OpenMP schedule\(dynamic, 64\)")):
                r[key] = fnum(lab + r": total ([\d.]+) s", blk)
            hs = re.findall(r"\((?:Table 3: [\d.]+\s+)(OK|MISMATCH)\)", blk)
            r["ok"] = len(hs) >= 9 and "MISMATCH" not in blk
            th = re.findall(r"^\s+(\d+)\s+(\d+)\s+([\d.]+)\s+(\d+)\s+([\d.]+)\s*$", blk, re.M)
            r["threads"] = [tuple(x) for x in th]
            r["imb"] = re.search(r"Link imbalance \(max/avg\): static ([\d.]+), dynamic ([\d.]+)", blk)
            r["total"] = r["serial"]
            if task == "all":
                corr.append(("2.1", b["p"], b["t"], r["n"], "OK" if r["ok"] else "MISMATCH"))
            else:
                t21[(r["n"], r["t"])].append(r)
        if task == "2.4" and "block boundaries" in tx:
            after = tx.split("block boundaries")[1].split("\n")[2:]
            rows_b = []
            for ln in after:
                f = ln.split()
                if len(f) != 5 or not all(x.isdigit() for x in f):
                    break
                rows_b.append(tuple(int(x) for x in f))
            bounds[b["p"]] = rows_b
        secs = q2_sections(tx)
        # Output of remote ranks can arrive a few lines late (after the next
        # section header), so also accept a run whose H lines are all OK overall.
        n_ok = len(re.findall(r"\(Table 3: [\d.]+\s+OK\)\s+\[printed by rank", tx))
        block_ok = "MISMATCH" not in tx and n_ok == 3 * len(secs)
        for s in secs:
            s["ok"] = s["ok"] or (block_ok and s["links_ok"])
            s["rep"] = b["rep"]
            if task == "all":
                corr.append((s["title"].split(" ")[1], s["p"], s["t"], s["n"],
                             "OK" if s["ok"] else "MISMATCH"))
            else:
                mpi[(s["title"], s["n"], s["p"], s["t"])].append(s)

    md.append("## Question 2 — Sparse matrix (social network)\n")
    md.append("### Q2 correctness, n = 100000 (H entries vs Table 3, 12 decimals, + total link count)\n")
    agg = defaultdict(list)
    for task, p, t, n, ok in corr:
        agg[(task, n)].append((p, ok))
    md.append(md_table(["task", "n", "process counts tested", "all OK"],
                       [(k[0], str(k[1]), ", ".join(str(p) for p in sorted({p for p, _ in v})),
                         "OK" if all(o == "OK" for _, o in v) else "MISMATCH")
                        for k, v in sorted(agg.items())]))
    allok = all(r["ok"] for v in t21.values() for r in v) and all(s["ok"] for v in mpi.values() for s in v)
    md.append(f"All benchmark runs (n = 1,000,000 etc.) print Table 3 values exactly: **{'YES' if allok else 'NO'}**\n\n")

    md.append("### Task 2.1 — CSR step, serial vs OpenMP (best of repeats; total for T = 10 steps, seconds)\n")
    rows = []
    for (n, t), runs in sorted(t21.items()):
        se = min(r["serial"] for r in runs)
        st = min(r["static"] for r in runs)
        dy = min(r["dynamic"] for r in runs)
        rows.append((str(n), str(t), str(len(runs)), se, st, dy, se / T, f"{se / st:.2f}", f"{se / dy:.2f}",
                     "OK" if all(r["ok"] for r in runs) else "MISMATCH"))
    md.append(md_table(["n", "threads", "runs", "serial", "static", "dynamic,64", "serial per step",
                        "speedup static", "speedup dynamic", "Table 3"], rows))
    md.append("#### Per-thread links per step, schedule(static) vs dynamic,64 (n = 1,000,000, one run)\n")
    for (n, t), runs in sorted(t21.items()):
        if n == 1000000 and t > 1:
            r = runs[0]
            md.append(f"threads = {t}" + (f" — link imbalance max/avg: static {r['imb'].group(1)}, "
                                          f"dynamic {r['imb'].group(2)}" if r["imb"] else "") + "\n\n")
            md.append(md_table(["thread", "static links", "static busy (s)", "dynamic links", "dynamic busy (s)"],
                               [(a, b_, c, d, e) for a, b_, c, d, e in r["threads"]]))

    def get(title_prefix, n, p, t=1):
        for (ti, nn, pp, tt), v in mpi.items():
            if ti.startswith(title_prefix) and nn == n and pp == p and tt == t:
                return best(v), len(v)
        return None, 0

    N = 1000000
    md.append("### Task 2.2 / 2.3 — Version A vs Version B (n = 1,000,000, best run)\n")
    md.append("recv/step = bytes received per process per step (max over ranks). "
              "comm includes wait.\n\n")
    rows = []
    for p in (1, 2, 4, 8):
        for pref, nm in (("Task 2.2", "A (Allgatherv)"), ("Task 2.3 Version B (halo, interior", "B overlap"),
                         ("Task 2.3 Version B (halo, all", "B no overlap")):
            s, k = get(pref, N, p)
            if not s:
                continue
            cm = max(r["compute"] for r in s["ranks"])
            co = max(r["comm"] for r in s["ranks"])
            wa = max(r["wait"] for r in s["ranks"])
            kib = max(r["kib"] for r in s["ranks"])
            rows.append((str(p), nm, str(k), s["total"], s["step"], cm, co, wa, f"{kib:.1f} KiB",
                         "OK" if s["ok"] else "MISMATCH"))
    md.append(md_table(["p", "version", "runs", "total (s)", "per step (s)", "compute max", "comm max",
                        "Waitall max", "recv/step", "Table 3"], rows))

    md.append("### Task 2.4 — block boundaries (equal rows vs equal links)\n")
    for p, rows_ in sorted(bounds.items()):
        md.append(f"p = {p}\n\n")
        md.append(md_table(["rank", "equal-rows first user", "rows", "equal-links first user", "rows"],
                           [tuple(str(x) for x in r) for r in rows_]))
    md.append("### Task 2.4 — equal rows vs equal links, Version B with overlap (best run)\n")
    rows = []
    for p, t in ((4, 1), (8, 1), (3, 1), (3, 2), (3, 4)):
        for pref, nm in (("Task 2.4 Version B, equal ROWS", "equal rows"),
                         ("Task 2.4 Version B, equal LINKS", "equal links")):
            s, k = get(pref, N, p, t)
            if not s:
                continue
            rows.append((f"{p} x {t}", nm, str(k), s["total"], s["step"], s["imb"],
                         max(r["compute"] for r in s["ranks"]), min(r["compute"] for r in s["ranks"]),
                         "OK" if s["ok"] else "MISMATCH"))
    md.append(md_table(["p x threads", "partition", "runs", "total (s)", "per step (s)",
                        "compute imbalance max/avg", "compute max", "compute min", "Table 3"], rows))

    md.append("### Q2 per-process computation / communication (best runs, n = 1,000,000; seconds over 10 steps)\n")
    rows = []
    for pref, nm, ps in (("Task 2.2", "A", (4, 8)), ("Task 2.3 Version B (halo, interior", "B overlap", (4, 8)),
                         ("Task 2.4 Version B, equal ROWS", "B equal rows", (8,)),
                         ("Task 2.4 Version B, equal LINKS", "B equal links", (8,))):
        for p in ps:
            s, _ = get(pref, N, p)
            if s:
                for r in s["ranks"]:
                    rows.append((nm, str(p), str(r["rank"]), str(r["rows"]), str(r["links"]),
                                 r["compute"], r["comm"], r["wait"]))
    md.append(md_table(["version", "p", "rank", "rows", "links", "compute", "comm", "wait"], rows))

    md.append("### Q2 summary — all versions side by side (n = 1,000,000, best runs)\n")
    rows = []
    ser = t21.get((N, 1))
    if ser:
        r0 = ser[0]
        base = min(r["serial"] for r in ser)
        rows.append(("serial (Code 3)", "1 x 1", base, "1.00"))
        for t in (1, 2, 4):
            if (N, t) in t21:
                d = min(r["dynamic"] for r in t21[(N, t)])
                rows.append((f"OpenMP dynamic,64", f"1 x {t}", d, f"{base / d:.2f}"))
        for pref, nm, cfgs in (("Task 2.2", "Version A", ((2, 1), (4, 1), (8, 1))),
                               ("Task 2.3 Version B (halo, interior", "Version B (overlap)", ((2, 1), (4, 1), (8, 1))),
                               ("Task 2.4 Version B, equal LINKS", "Version B balanced", ((4, 1), (8, 1), (3, 1), (3, 4)))):
            for p, t in cfgs:
                s, _ = get(pref, N, p, t)
                if s:
                    rows.append((nm + (" hybrid" if p == 3 else ""), f"{p} x {t}", s["total"],
                                 f"{base / s['total']:.2f}"))
        md.append("H values printed (identical in every run): " +
                  ", ".join(f"H({a},{b})={v}" for a, b, v, _ in
                            (best(mpi[next(iter(mpi))])["H"] if mpi else [])) + "\n\n")
    md.append(md_table(["version", "p x threads", "total time T=10 (s)", "speedup vs serial"], rows))
    return t21, mpi


T = 10


# ============================================================================
# Q3
def q3(md):
    bl = blocks(os.path.join(RES, "q3_outputs.txt"))
    if not bl:
        md.append("_results/q3_outputs.txt not found_\n")
        return
    runs = defaultdict(list)
    bench = {}
    corr = []
    for b in bl:
        if not b["args"]:
            continue
        m_, task = int(b["args"][0]), b["args"][1]
        tx = b["text"]
        # split into training runs: each starts with Table 4(a) and ends after the 4(b) lines
        pieces = re.split(r"\n(?=\s*Table 4\(a\))", "\n" + tx)
        for pc in pieces:
            h = re.search(r"--- (Task 3\.[^:]+): m=(\d+), p=(\d+), threads=(\d+) ---", pc)
            if not h:
                continue
            a = re.search(r"Table 4\(a\) m=\d+: G0\(1,2\)=(-?\d+)\s+G0\(\d+,\d+\)=(-?\d+)\s+G7\(\d+,\d+\)=(-?\d+)", pc)
            ga = [int(x) for x in a.groups()] if a else None
            wb = re.findall(r"Table 4\(b\) m=\d+: W0\(0,0\)=(\S+)\s+W0\(1,2\)=(\S+)\s+W7\(\d+,\d+\)=(\S+)", pc)
            m = int(h.group(2))
            ok = ga == TABLE4A.get(m) and len(wb) >= 1 and all(list(w) == TABLE4B.get(m) for w in wb)
            prr = re.findall(r"^\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s*$",
                             pc.split("Per-process times per step")[-1], re.M) if "Per-process" in pc else []
            r = dict(title=h.group(1), m=m, p=int(h.group(3)), t=int(h.group(4)), rep=b["rep"],
                     total=fnum(r"Total:\s+([\d.]+)", pc), comp=fnum(r"Compute:\s+([\d.]+)", pc),
                     ar=fnum(r"Allreduce:\s+([\d.]+)", pc), ga=ga, wb=wb, nwb=len(wb), ok=ok,
                     ranks=[(int(x), float(y), float(z)) for x, y, z in prr])
            if m == 512:
                corr.append(r)
            else:
                runs[(r["title"], m, r["p"], r["t"])].append(r)
        bm = re.search(r"all-reduce benchmark: p=(\d+) ---\n.*\n", tx)
        if bm:
            rows_b = []
            for ln in tx[bm.end():].split("\n"):
                f = ln.split()
                if len(f) < 2 or not re.match(r"^[\d.]+$", f[1]):
                    break
                rows_b.append(f + ["(not finished)"] * (4 - len(f)))
            bench[int(bm.group(1))] = rows_b

    md.append("## Question 3 — All-reduce in parallel training\n")
    md.append("### Q3 correctness, m = 512 (Table 4(a) on rank 0, Table 4(b) printed by every rank)\n")
    md.append(md_table(["version", "p", "G0(1,2)", "G0(511,511)", "G7(511,511)", "ranks printing 4(b)",
                        "W0(0,0)", "W0(1,2)", "W7(511,511)", "Table 4"],
                       [(r["title"], str(r["p"]), *[str(x) for x in (r["ga"] or ["?"] * 3)], str(r["nwb"]),
                         *(r["wb"][0] if r["wb"] else ["?"] * 3), "OK" if r["ok"] else "MISMATCH")
                        for r in corr]))
    # timing of the m = 512 runs (the only Q3 timings if the m = 1024 runs were not done)
    g512 = defaultdict(list)
    for r in corr:
        g512[(r["title"], r["p"], r["t"])].append(r)
    md.append("### Q3 timing, m = 512 (best run; per-step times are max over ranks)\n")
    rows = []
    t1 = {}
    for (ti, p, t), v in sorted(g512.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        r = best(v)
        if p == 1:
            t1[ti] = r["total"]
        sp = t1.get(ti)
        rows.append((ti, f"{p} x {t}", str(len(v)), r["total"], r["comp"], r["ar"],
                     f"{100 * r['ar'] * 5 / r['total']:.0f}%",
                     f"{sp / r['total']:.2f}" if sp else "–", f"{sp / r['total'] / p:.2f}" if sp else "–"))
    md.append(md_table(["version", "p x threads", "runs", "total S=5 (s)", "compute/step", "all-reduce/step",
                        "all-reduce share", "speedup T(1)/T(p)", "efficiency"], rows))
    md.append("### Q3 per-process compute / all-reduce time per step (m = 512)\n")
    rows = []
    for (ti, p, t), v in sorted(g512.items()):
        if p in (4, 8):
            r = best(v)
            for rk, c, a in r["ranks"]:
                rows.append((ti, str(p), str(rk), c, a))
    md.append(md_table(["version", "p", "rank", "compute/step (s)", "all-reduce/step (s)"], rows))

    allok = all(r["ok"] for v in runs.values() for r in v)
    if not runs:
        md.append("**No m = 1024 runs in the log** (Tasks 3.1/3.2/3.3 at m = 1024 were not run).\n\n")
        bench_only = True
    else:
        bench_only = False
    if not bench_only:
        md.append(f"All m = 1024 runs print Table 4(a)/(b) exactly on every rank: **{'YES' if allok else 'NO'}**\n\n")

        md.append("### Task 3.1 / 3.2 / 3.3 — m = 1024 training (best run by total; per-step times are max over ranks)\n")
        rows = []
        base = None
        for (ti, m, p, t), v in sorted(runs.items(), key=lambda kv: (kv[0][0], kv[0][2], kv[0][3])):
            r = best(v)
            rows.append((ti, f"{p} x {t}", str(len(v)), r["total"], r["comp"], r["ar"],
                         f"{100 * r['ar'] * 5 / r['total']:.0f}%", *[str(x) for x in r["ga"]],
                         r["wb"][0][2] if r["wb"] else "?", "OK" if all(x["ok"] for x in v) else "MISMATCH"))
        md.append(md_table(["version", "p x threads", "runs", "total S=5 (s)", "compute/step", "all-reduce/step",
                            "all-reduce share", "G0(1,2)", "G0(1023,1023)", "G7(1023,1023)", "W7(1023,1023)",
                            "Table 4"], rows))

        md.append("### Task 3.3 — scaling with MPI_Allreduce, m = 1024, batch 512 (best runs)\n")
        rows = []
        sc = {(p, t): best(v) for (ti, m, p, t), v in runs.items() if ti.startswith("Task 3.3")}
        if (1, 1) in sc:
            t1 = sc[(1, 1)]["total"]
            for (p, t), r in sorted(sc.items()):
                cores = min(p, 3)
                rows.append((f"{p} x {t}", r["total"], r["comp"], r["ar"], f"{t1 / r['total']:.2f}",
                             f"{t1 / r['total'] / p:.2f}", f"{t1 / r['total'] / cores:.2f}"))
        md.append(md_table(["p x threads", "total (s)", "compute/step", "all-reduce/step", "speedup T(1)/T(p)",
                            "efficiency (/p)", "efficiency (/physical cores)"], rows))

    md.append("### Task 3.2 — all-reduce benchmark (seconds per call, mean of 50 calls)\n")
    rows = []
    for p, lines in sorted(bench.items()):
        for ln in lines:
            rows.append((str(p), ln[0], *[float(x) if re.match(r"^[\d.]+$", x) else x for x in ln[1:4]]))
    md.append(md_table(["p", "method", "1 KB", "1 MB", "64 MB"], rows))

    if bench_only:
        return
    md.append("### Q3 per-process compute / all-reduce time per step (best runs, m = 1024)\n")
    rows = []
    for (ti, m, p, t), v in sorted(runs.items()):
        if p in (4, 8) and t == 1 and not ti.startswith("Task 3.3"):
            r = best(v)
            for rk, c, a in r["ranks"]:
                rows.append((ti, str(p), str(rk), c, a))
    md.append(md_table(["version", "p", "rank", "compute/step (s)", "all-reduce/step (s)"], rows))


def main():
    md = ["# Results summary (generated by summarize_results.py)\n",
          "Best-of-repeats tables built from results/q1_outputs.txt, q2_outputs.txt and "
          "q3_outputs.txt. 'runs' = how many repeats were found for that configuration.\n\n"]
    q1(md)
    q2(md)
    q3(md)
    out = os.path.join(RES, "summary.md")
    open(out, "w").write("".join(md))
    print("wrote", out)


if __name__ == "__main__":
    sys.exit(main())
