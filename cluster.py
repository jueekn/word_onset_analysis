"""cluster.py — small helpers for monitoring dask futures (notebook-driven).

Used by data_visualization / data_validation / unit_tests / sim notebooks
to poll a dask client for progress + checkpoint partial results. Not part
of the snakemake-driven pipeline (run_project.sh).

Dominated by IPython.display + cmldask + dask method chains; narrow the
four library-stub-noise rules at file scope.
"""
# pyright: reportUnknownMemberType=warning, reportUnknownArgumentType=warning, reportUnknownVariableType=warning, reportMissingTypeStubs=none
from __future__ import annotations

from typing import Any, Callable
import pickle

from IPython.display import display, clear_output
from cmldask import CMLDask
from time import sleep, time
from cmldask.CMLDask import get_exceptions


def get_exceptions_quiet(futures: Any, params: Any) -> Any:
    """Return cmldask.get_exceptions output, swallowing the no-exceptions error.

    cmldask raises rather than returning empty when no futures errored; this
    wrapper returns None in that case so the caller can branch cleanly.
    """
    try:
        exceptions = get_exceptions(futures, params)
        print('Exceptions occurred during cluster run!')
        return exceptions
    except Exception:
        return None


def wait(
    futures: Any,
    client: Any,
    check_delay: float = 10,
    cancel_prop: float = 1.0,
    checkpoint_file: str | None = None,
    visualize_func: Callable[[Any], Any] | None = None,
) -> None:
    """Block until every future in `futures` is done, polling every check_delay s.

    Side effects per poll: clears the notebook output cell, prints progress +
    error counts, optionally pickles partial results to `checkpoint_file`,
    optionally displays a notebook-friendly `visualize_func(results)` view.
    """
    assert isinstance(check_delay, (float, int))
    start = time()
    while True:
        sleep(check_delay)
        clear_output()
        # TODO: should probably reserve full result gathering if visualize_func is specified to reduce network load/overhead
        finished_futures = CMLDask.filter_futures(futures)
        n_finished = len(finished_futures)

        # check for errors and cancel simulation if over cancel_prop proportion of jobs are errors
        errors = None
        try:  # CMLDask throws an error if there were no exceptions in any Dask jobs
            errors = CMLDask.get_exceptions(futures, range(len(futures)))
            print("Dask Errors:")
            print(errors.head())
            n_errors = len(errors)
        except Exception:
            n_errors = 0

        dur = time() - start
        rate = -1.0 if not n_finished else n_finished/dur
        print(f'Simulations finished after {dur:0.3} s: {n_finished + n_errors} / {len(futures)} ({rate:0.3} iterations/s). {n_errors} job errors')

        results: Any = None
        if checkpoint_file or (visualize_func is not None):
            results = client.gather(finished_futures)

        if visualize_func is not None:
            display(visualize_func(results))

        if checkpoint_file:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(results, f)

        if n_finished + n_errors == len(futures):
            print('Simulation complete. Shutting down jobs.')
            break
