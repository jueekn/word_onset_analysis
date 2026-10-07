# prepare_sessions -> compute.py (power + synchrony, one load per session) -> plot_*.py.
# Per-session caching stays inside compute.py; the rules only order them.
#   snakemake --cores 8                                          # full run
#   snakemake --cores 4 --config smokescreen=true                # N subjects, *.smokescreen outputs
#   snakemake simulations --cores 4 [--config smokescreen=true]  # validity checks -> figures/simulations/
#   add longetal=true for the Long et al. 2020 replication (power only, build_roi_power.py) -> *.longetal
import os, sys

configfile: "config/config.yaml"
workdir: workflow.basedir

SMOKE = str(config["smokescreen"]).lower() == "true"
LONG = str(config.get("longetal", False)).lower() == "true"
if SMOKE:
    os.environ["WOA_SMOKESCREEN"] = "1"   # project_paths: scratch + exclusion logs -> *.smokescreen
if LONG:
    os.environ["WOA_LONGETAL"] = "1"      # project_paths: Long et al. settings, scratch -> *.longetal
os.environ["CML_AUTO_APPROVE"] = "1"      # cml_data would otherwise prompt before downloading

from project_paths import SCRATCH_DIR
from simulate_eeg import SWEEPS

PY = sys.executable   # the env snakemake runs in
FIG = "figures" + (".longetal" if LONG else "") + (".smokescreen" if SMOKE else "")
DONE = f"{FIG}/.done"
SWEEP_TAGS = [t for tags in SWEEPS.values() for t in tags]   # recovery sweeps: compute only
# plot script -> its figure subdirectory ("" = the script writes its own subdirectories)
PLOTS = {"plot_boxplots": "", "plot_spectrum": "burke_roi_power", "plot_case_studies": "burke_roi_power",
         "plot_timecourse": "burke_roi_power", "plot_epoch_network": "burke_roi_synchrony",
         "plot_latency": "latency"}


def sim(w): return "" if w.sim == "real" else f"--simulation-tag {w.sim}"   # sims -> <scratch>/sim/<tag>
def fig(w): return FIG if w.sim == "real" else f"{FIG}/simulations/{w.sim}"
def plot_cmds(w):
    return "; ".join(f"{PY} {s}.py {sim(w)} --out-dir {os.path.join(fig(w), d)}".rstrip("/") for s, d in PLOTS.items())


rule all:
    input: expand(f"{DONE}/real/{{band}}.long_power", band=[r["band"] for r in config["longetal_params"]["runs"]]) if LONG
           else f"{DONE}/real/plots"

rule simulations:
    input: expand(f"{DONE}/{{sim}}/plots", sim=config["simulations"]), f"{FIG}/simulations/recovery.png"

rule recovery:
    input: expand(f"{DONE}/{{sim}}/compute", sim=SWEEP_TAGS)
    output: f"{FIG}/simulations/recovery.png"
    shell: f"{PY} plot_recovery.py --out-dir {FIG}/simulations"

rule prepare_sessions:
    output: f"{SCRATCH_DIR}/sess_list_df.json"
    threads: workflow.cores
    params: n=f"--n-subjects {config['smokescreen_n_subjects']}" if SMOKE else ""
    shell: f"{PY} prepare_sessions.py --workers {{threads}} {{params.n}}"

# threads = all cores, so compute runs one at a time (RAM, shared downloads).
rule compute:
    input: rules.prepare_sessions.output
    output: touch(f"{DONE}/{{sim}}/compute")
    threads: workflow.cores
    params: sim=sim
    shell: f"{PY} compute.py {{params.sim}} --workers {{threads}}"

rule plots:
    input: f"{DONE}/{{sim}}/compute"
    output: touch(f"{DONE}/{{sim}}/plots")
    params: cmds=plot_cmds
    shell: "{params.cmds}"

rule long_power:   # Long et al. replication (closed): legacy compute + plot
    input: rules.prepare_sessions.output
    output: touch(f"{DONE}/real/{{band}}.long_power")
    threads: workflow.cores
    shell: f"{PY} build_roi_power.py --band {{wildcards.band}} --workers {{threads}} --out-dir {FIG}/burke_roi_power"
