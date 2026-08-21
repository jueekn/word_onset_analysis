# pyright: reportUnknownVariableType=warning, reportUnknownMemberType=warning, reportUnknownArgumentType=warning, reportMissingTypeStubs=none
"""Project wrapper around cmldask's SLURM dask client.

Every project dispatch path MUST use `project_dask_client` instead of importing
`cmldask.CMLDask.new_dask_client_slurm` directly. It pins the two directories
cmldask otherwise defaults to `$HOME` (the chronically-full /home1 NFS):

  - worker scratch / spill (`local_directory`) -> SCRATCH_DIR/dask_worker_space/<job_name>
  - worker logs          (`log_directory`)     -> LOGS_DIR/dask/<job_name>

The per-job subdir slots in automatically from the `job_name` callers already
pass — no new rule/script parameters. Concurrent clusters get isolated subdirs,
so nothing collides and nothing lands in $HOME.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from cmldask.CMLDask import new_dask_client_slurm  # pyright: ignore[reportMissingTypeStubs]

from project_paths import LOGS_DIR, SCRATCH_DIR


def project_dask_client(
    job_name: str,
    memory_per_job: str,
    max_n_jobs: int,
    *,
    walltime: int | str,
    queue: str = "RAM,RAM-GPU",
    job_extra_directives: list[str] | None = None,
    **kwargs: Any,
) -> Any:
    """cmldask SLURM client with scratch+logs routed to project dirs (never $HOME)."""
    local_dir = Path(SCRATCH_DIR) / "dask_worker_space" / job_name
    log_dir = Path(LOGS_DIR) / "dask" / job_name
    local_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    return new_dask_client_slurm(
        job_name,
        memory_per_job,
        max_n_jobs,
        walltime=walltime,  # pyright: ignore[reportArgumentType]
        queue=queue,
        local_directory=str(local_dir),
        log_directory=str(log_dir),
        job_extra_directives=list(job_extra_directives or []),
        **kwargs,
    )
