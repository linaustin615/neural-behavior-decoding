"""Keep the exact interleaved decoder; vary only readout key input."""
import importlib.util
from pathlib import Path

source = Path(__file__).resolve().parent.parent / '2026-10-05_temporal_specialization' / 'models.py'
spec = importlib.util.spec_from_file_location('interleaved_parent', source)
parent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent)


class InterleavedDecoder(parent.SpecializedDecoder):
    def __init__(self, variant, seed, sessions):
        assert variant in ['dynamic', 'static']
        super().__init__('interleaved_dynamic', seed, sessions)
        self.variant = variant
        self.kind = 'attention' if variant == 'dynamic' else 'mlp'
