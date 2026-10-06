from torch import nn

class NeuronTokenizer(nn.Module):
    def __init__(self, window=8, width=32, n_neurons=128):
        super().__init__()

        self.activity_embedding = nn.Linear(window, width)

        self.position_embedding = nn.Sequential(
            nn.Linear(3, width),
            nn.GELU(),
            nn.Linear(width, width),
        )

        self.identity_embedding = nn.Embedding(n_neurons, width)
        nn.init.normal_(self.identity_embedding.weight, std=0.02)

    def forward(self, activity, positions, token_ids):
        activity_tokens = self.activity_embedding(activity)
        position_tokens = self.position_embedding(positions)
        identity_tokens = self.identity_embedding(token_ids)

        return activity_tokens + position_tokens + identity_tokens #(batch_size, n_neurons, width)


class NeuronEncoder(nn.Module):
    def __init__(self, window=8, width=32, n_neurons=128):
        super().__init__()
        self.tokenizer = NeuronTokenizer(window, width, n_neurons)
        self.transformer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=4,
            dim_feedforward=width * 2,
            dropout=0.05,
            batch_first=True,
        )

    def forward(self, activity, positions, token_ids):
        tokens = self.tokenizer(activity, positions, token_ids) #32-feature tokens
        return self.transformer(tokens) #use attention to combine information across neurons


class RunningSpeedDecoder(nn.Module):
    def __init__(self, window=8, width=32, n_neurons=128):
        super().__init__()
        self.encoder = NeuronEncoder(window, width, n_neurons)
        self.head = nn.Linear(width, 1)

    def forward(self, activity, positions, token_ids):
        encoded = self.encoder(activity, positions, token_ids)
        pooled = encoded.mean(dim=1)
        return self.head(pooled).squeeze(-1)


class LinearSpeedDecoder(nn.Module):
    def __init__(self, window=8, n_neurons=128):
        super().__init__()
        self.head = nn.Linear(n_neurons * window, 1)

    def forward(self, activity, positions, token_ids):
        flattened = activity.flatten(start_dim=1)
        return self.head(flattened).squeeze(-1)

class NoPositionSpeedDecoder(RunningSpeedDecoder):
    def forward(self, activity, positions, token_ids):
        return super().forward(activity, positions * 0, token_ids)