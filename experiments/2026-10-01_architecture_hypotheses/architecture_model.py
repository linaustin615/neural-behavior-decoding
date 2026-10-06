import torch
from torch import nn
import base_search as s


class ArchitectureCandidate(s.Candidate):
    def __init__(self, config):
        shared_config = dict(config, latents=8)
        super().__init__(shared_config)
        self.config = dict(config)
        self.variant = config['variant']
        seed = config['initialization_seed']
        if self.variant == 'latent32':
            generator = torch.Generator().manual_seed(700000 + seed)
            extra = torch.randn(1, 24, config['width'], generator=generator) * .02
            self.latents = nn.Parameter(torch.cat([self.latents.detach(), extra], dim=1))
        if self.variant == 'no_id':
            self.base.encoder.tokenizer.identity_embedding.weight.requires_grad_(False)
        if self.variant == 'local':
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(800000 + seed)
                self.local_embedding = nn.Sequential(
                    nn.Linear(config['window'] * 2, config['width']),
                    nn.GELU(),
                    nn.Linear(config['width'], config['width']),
                )
        self.register_buffer('neighbor_graph', None, persistent=False)
        self._geometry_key = None
        self._geometry_positions = None
        self.graph_info = None

    def prepare_geometry(self, positions):
        if self.variant != 'local':
            return
        key = (positions.data_ptr(), positions._version, tuple(positions.shape))
        if positions is self._geometry_positions and key == self._geometry_key:
            return
        n = len(positions)
        assert n > 1
        with torch.no_grad():
            if torch.count_nonzero(positions) == 0:
                self.neighbor_graph = None
                self.graph_info = {'kind': 'all_other_cells_mean', 'neighbors': n - 1}
            else:
                distances = torch.cdist(positions, positions)
                distances.fill_diagonal_(float('inf'))
                k = min(8, n - 1)
                values, neighbors = distances.topk(k, largest=False, dim=1)
                bandwidth = values[:, -1].median().clamp_min(1e-6)
                weights = torch.exp(-.5 * (values / bandwidth).square())
                weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-20)
                rows = torch.arange(n).repeat_interleave(k)
                indices = torch.stack([rows, neighbors.flatten()])
                self.neighbor_graph = torch.sparse_coo_tensor(indices, weights.flatten(), (n, n)).coalesce()
                self.graph_info = {'kind': 'gaussian_knn', 'neighbors': k, 'bandwidth': float(bandwidth)}
        self._geometry_key = key
        self._geometry_positions = positions

    def neighborhood(self, activity, positions):
        self.prepare_geometry(positions)
        batch, n, window = activity.shape
        if self.neighbor_graph is None:
            return (activity.sum(dim=1, keepdim=True) - activity) / (n - 1)
        flat = activity.permute(1, 0, 2).reshape(n, batch * window)
        return torch.sparse.mm(self.neighbor_graph, flat).reshape(n, batch, window).permute(1, 0, 2)

    def forward(self, activity, positions, token_ids):
        tokenizer = self.base.encoder.tokenizer
        tokens = tokenizer.activity_embedding(activity) + tokenizer.position_embedding(positions)
        if self.variant != 'no_id':
            tokens = tokens + tokenizer.identity_embedding(token_ids)
        if self.variant == 'local':
            neighbor_activity = self.neighborhood(activity, positions)
            local_input = torch.cat([activity, neighbor_activity - activity], dim=-1)
            tokens = tokens + .1 * self.local_embedding(local_input)
        queries = self.latents.expand(activity.shape[0], -1, -1)
        latent, _ = self.readin(queries, tokens, tokens, need_weights=False)
        encoded = self.base.encoder.transformer(latent + queries)
        return self.base.head(encoded.mean(dim=1)).squeeze(-1)
