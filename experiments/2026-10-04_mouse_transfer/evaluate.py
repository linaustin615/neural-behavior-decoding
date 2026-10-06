"""Run the frozen evaluator with its missing metadata-only import supplied.

The completed study must not be scored again. This entry point is for a clean
reproduction after preparing and locking new fits; the frozen evaluator refuses
to overwrite an existing results.json.
"""
import importlib.util
from pathlib import Path
import platform

from threadpoolctl import threadpool_limits


ROOT=Path(__file__).resolve().parent


if __name__=='__main__':
    spec=importlib.util.spec_from_file_location('frozen_transfer_evaluator',ROOT/'run.py')
    frozen=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    frozen.platform=platform
    with threadpool_limits(limits=2):
        frozen.torch.set_num_threads(2)
        frozen.torch.set_num_interop_threads(1)
        frozen.evaluate()
