"""Separate temporal mixing from activity-dependent behavior-query pooling."""
import importlib.util
from pathlib import Path

source=Path(__file__).resolve().parent.parent/'2026-10-03_shared_behavior'/'models.py'
spec=importlib.util.spec_from_file_location('archived_behavior_components',source)
archived=importlib.util.module_from_spec(spec);spec.loader.exec_module(archived)
BehaviorDecoder=archived.BehaviorDecoder

VARIANTS={
    'aa':('attention','attention'),
    'as':('attention','mlp'),
    'ma':('mlp','attention'),
    'ms':('mlp','mlp'),
}


class FactorialDecoder(BehaviorDecoder):
    def __init__(self,variant,seed,sessions):
        temporal,query=VARIANTS[variant]
        super().__init__(temporal,seed,sessions)
        #the inherited forward uses kind only to choose activity or static routing keys
        self.kind=query
        self.temporal_kind=temporal
        self.query_kind=query
        self.variant=variant
