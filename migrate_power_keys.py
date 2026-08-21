"""Rename log10_pre/post -> log10_lo/hi in existing power pickles. Idempotent."""
import sys, glob, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fc_comparison_functions as fc

d = sys.argv[1]
n_done = n_skip = 0
for f in sorted(glob.glob(f"{d}/*_power.pkl")):
    p = fc.load_pickle(f)
    if "log10_lo" in p:
        n_skip += 1
        continue
    if "log10_pre" not in p or "n_win_samples" not in p:
        print(f"[leave] {f.split('/')[-1]}: not a migratable pickle")
        continue
    p["log10_lo"] = p.pop("log10_pre")
    p["log10_hi"] = p.pop("log10_post")
    p["lo_win"] = p.pop("pre_win")
    p["hi_win"] = p.pop("post_win")
    # prepost behaviors use the same events for both arms
    p["n_lo"] = p["n_hi"] = int(p["n_events"])
    fc.save_pickle(f, p)
    n_done += 1
print(f"migrated {n_done}, already current {n_skip}")
