"""The accepted shared decoder with a fixed 512-neuron panel."""
import importlib.util
from pathlib import Path

source=Path(__file__).resolve().parent.parent/'2026-10-03_shared_behavior'/'models.py'
spec=importlib.util.spec_from_file_location('larger_panel_decoder',source)
native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
native.NEURONS=512
BehaviorDecoder=native.BehaviorDecoder
