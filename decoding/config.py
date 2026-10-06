"""The fixed recipes used for the separate-recording comparison."""
from dataclasses import asdict, dataclass, replace


MICE = ('TX103', 'TX104', 'TX56', 'TX57', 'TX60', 'TX61', 'VR2')
SEEDS = (401, 402, 403)
NEURONS = 512
WARMUP = 63
GAP = 64
BATCH_SIZE = 64
RIDGE_HISTORIES = (16, 32, 64)
RIDGE_PENALTIES = (0.0001, 0.001, 0.01, 0.1, 1., 10., 100., 1000.)


@dataclass(frozen=True)
class Recipe:
    family: str = 'attention'
    width: int = 64
    depth: int = 2
    history: int = 32
    patch: int = 4
    dropout: float = 0.
    lr: float = 0.001
    weight_decay: float = 0.001
    epochs: int = 48

    def to_dict(self):
        return asdict(self)


RECIPES = {
    'transformer': Recipe(),
    'mlp': Recipe(family='mlp', patch=2, dropout=0.1),
    'small_transformer': Recipe(width=16, depth=1, dropout=0.05, weight_decay=0.01, epochs=24),
}


def recipe(name, epochs=None):
    result = RECIPES[name]
    if epochs is not None:
        if epochs < 1:
            raise ValueError('Epoch count must be positive.')
        result = replace(result, epochs=epochs)
    return result
