# prepare_sessions -> ROI power + ROI synchrony -> power-synchrony, for each entry
# in config.yaml `runs`. Per-session caching stays inside the scripts; the rules
# only order them.
#   snakemake --cores 8                                          # full run
#   snakemake --cores 4 --config smokescreen=true                # N subjects, *.smokescreen outputs
#   snakemake simulations --cores 4 [--config smokescreen=true]  # validity checks -> figures/simulations/
#   add longetal=true to any of these: Long et al. 2020 replication (power only) -> *.longetal
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
from simulate_eeg import SWEEPS, simulation_parameters

PY = sys.executable   # the env snakemake runs in
FIG = "figures" + (".longetal" if LONG else "") + (".smokescreen" if SMOKE else "")
LCFG = config["longetal_params"] if LONG else config
LAST = "power" if LONG else "power_synchrony"   # Long ran no synchrony
DONE = f"{FIG}/.done"
RUNS = {f"{r['band']}_{r['metric']}": r for r in LCFG["runs"]}
ARGS = "--band {params.r[band]} {params.sim} {params.stage}"
SWEEP_TAGS = [t for tags in SWEEPS.values() for t in tags     # recovery sweeps: compute only
              if not (LONG and simulation_parameters[t].get("data_generating_process") != "hg_power")]
RUN0 = next(k for k, r in RUNS.items() if r["band"] == "high_gamma")   # the sweeps plant high gamma


def r(w): return RUNS[w.run]
def sim(w): return "" if w.sim == "real" else f"--simulation-tag {w.sim}"   # sims -> <scratch>/sim/<tag>
def fig(w): return FIG if w.sim == "real" else f"{FIG}/simulations/{w.sim}"
def stage(w): return "--stage compute" if w.sim in SWEEP_TAGS else ""
def metrics(w): return r(w)["metric"] + ("" if w.sim == "real" else " ppc")   # recovery reads PPC (phase plant)
def rec_input(t): return f"{DONE}/{t}/{RUN0}." + ("power" if simulation_parameters[t].get("data_generating_process") == "hg_power" else "synchrony")


rule all:
    input: expand(f"{DONE}/real/{{run}}.{LAST}", run=RUNS)

rule simulations:
    input: expand(f"{DONE}/{{sim}}/{{run}}.{LAST}", sim=LCFG["simulations"], run=RUNS),
           f"{FIG}/simulations/recovery.png"

rule recovery:
    input: [rec_input(t) for t in SWEEP_TAGS]
    output: f"{FIG}/simulations/recovery.png"
    shell: f"{PY} plot_recovery.py --out-dir {FIG}/simulations"

rule prepare_sessions:
    output: f"{SCRATCH_DIR}/sess_list_df.json"
    threads: workflow.cores
    params: n=f"--n-subjects {config['smokescreen_n_subjects']}" if SMOKE else ""
    shell: f"{PY} prepare_sessions.py --workers {{threads}} {{params.n}}"

# threads = all cores, so compute rules run one at a time (RAM, shared downloads).
rule roi_power:
    input: rules.prepare_sessions.output
    output: touch(f"{DONE}/{{sim}}/{{run}}.power")
    threads: workflow.cores
    params: r=r, sim=sim, fig=fig, stage=stage
    shell: f"{PY} build_roi_power.py {ARGS} --workers {{threads}} --out-dir {{params.fig}}/burke_roi_power"

rule roi_synchrony:   # after power: power -> synchrony -> power-synchrony
    input: rules.prepare_sessions.output, f"{DONE}/{{sim}}/{{run}}.power"
    output: touch(f"{DONE}/{{sim}}/{{run}}.synchrony")
    threads: workflow.cores
    params: r=r, sim=sim, fig=fig, stage=stage, m=metrics
    shell: f"{PY} build_roi_synchrony.py {ARGS} --metrics {{params.m}} --metric {{params.r[metric]}} --workers {{threads}} --out-dir {{params.fig}}/burke_roi_synchrony"

rule power_synchrony:
    input: f"{DONE}/{{sim}}/{{run}}.power", f"{DONE}/{{sim}}/{{run}}.synchrony"
    output: touch(f"{DONE}/{{sim}}/{{run}}.power_synchrony")
    params: r=r, sim=sim, fig=fig, stage=stage
    shell: f"{PY} build_power_synchrony.py {ARGS} --metric {{params.r[metric]}} --out-dir {{params.fig}}/power_synchrony"
