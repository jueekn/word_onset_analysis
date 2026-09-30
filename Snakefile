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
RUNS = {"_".join(r[k] for k in ("beh", "band", "fc_mode", "metric")): r for r in LCFG["runs"]}
ARGS = "--beh {params.r[beh]} --band {params.r[band]} --fc-mode {params.r[fc_mode]} {params.sim} {params.stage}"
CONTRAST_ONLY = not LONG   # juee figures: drop the word-on / word-off panels, keep the contrast
SWEEP_TAGS = [t for tags in SWEEPS.values() for t in tags     # recovery sweeps: compute only
              if not (LONG and simulation_parameters[t].get("data_generating_process") != "hg_power")]
RUN0 = next(iter(RUNS))                                        # recovery uses the first run


def r(w): return RUNS[w.run]
def sim(w): return "" if w.sim == "real" else f"--simulation-tag {w.sim}"   # sims -> <scratch>/sim/<tag>
def fig(w): return FIG if w.sim == "real" else f"{FIG}/simulations/{w.sim}"
def stage(w): return "--stage compute" if w.sim in SWEEP_TAGS else ""
def rec_input(t): return f"{DONE}/{t}/{RUN0}." + ("power" if simulation_parameters[t].get("data_generating_process") == "hg_power" else "synchrony")


rule all:
    input: expand(f"{DONE}/real/{{run}}.{LAST}", run=RUNS),
           [] if LONG else f"{FIG}/band_contrasts/power_word_on.png"

# juee: word on - word off only, alpha and high gamma in one figure per analysis
rule band_contrasts:
    input: expand(f"{DONE}/real/{{run}}.power_synchrony", run=RUNS)
    output: f"{FIG}/band_contrasts/power_word_on.png"
    shell: f"{PY} plot_band_contrasts.py --fig-dir {FIG}"

rule simulations:
    input: expand(f"{DONE}/{{sim}}/{{run}}.{LAST}", sim=LCFG["simulations"], run=RUNS),
           f"{FIG}/simulations/recovery.png"

rule recovery:
    input: [rec_input(t) for t in SWEEP_TAGS]
    output: f"{FIG}/simulations/recovery.png"
    shell: f"{PY} plot_recovery.py --out-dir {FIG}/simulations --fc-mode {RUNS[RUN0]['fc_mode']}"

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
    shell: f"{PY} build_roi_power.py {ARGS} {'--measures cohens_d' if CONTRAST_ONLY else ''} --workers {{threads}} --out-dir {{params.fig}}/burke_roi_power"

rule roi_synchrony:   # after power: power -> synchrony -> power-synchrony
    input: rules.prepare_sessions.output, f"{DONE}/{{sim}}/{{run}}.power"
    output: touch(f"{DONE}/{{sim}}/{{run}}.synchrony")
    threads: workflow.cores
    params: r=r, sim=sim, fig=fig, stage=stage
    shell: f"{PY} build_roi_synchrony.py {ARGS} --metrics {{params.r[metric]}} --metric {{params.r[metric]}} {'--measures sync_diff' if CONTRAST_ONLY else ''} --workers {{threads}} --out-dir {{params.fig}}/burke_roi_synchrony"

rule power_synchrony:
    input: f"{DONE}/{{sim}}/{{run}}.power", f"{DONE}/{{sim}}/{{run}}.synchrony"
    output: touch(f"{DONE}/{{sim}}/{{run}}.power_synchrony")
    params: r=r, sim=sim, fig=fig, stage=stage
    shell: f"{PY} build_power_synchrony.py {ARGS} --metric {{params.r[metric]}} {'--conds diff' if CONTRAST_ONLY else ''} --out-dir {{params.fig}}/power_synchrony"
