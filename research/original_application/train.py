import torch
from torch import nn
from copy import deepcopy

from data import train_loader, val_loader, positions_tensor, y_train, y_val
from model import RunningSpeedDecoder, LinearSpeedDecoder, NoPositionSpeedDecoder

token_ids = torch.arange(positions_tensor.shape[0])

speed_mean = float(y_train.mean())
speed_std = max(float(y_train.std()), 1e-6)

normalized_val_speed = (y_val - speed_mean) / speed_std
baseline_val_loss = float((normalized_val_speed ** 2).mean())

print(f"constant-speed baseline: {baseline_val_loss:.4f}")

criterion = nn.MSELoss()

def evaluate(model, loader):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for activity_batch, speed_batch in loader:
            targets = (speed_batch - speed_mean) / speed_std
            predictions = model(activity_batch, positions_tensor, token_ids)
            loss = criterion(predictions, targets)
            total_loss += loss.item() * len(targets)

    return total_loss / len(loader.dataset)

def train_model(model_class, seed, num_epochs=10):
    torch.manual_seed(seed) #randomness
    train_loader.generator.manual_seed(seed) #training batch order

    model = model_class()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    best_val_loss = evaluate(model, val_loader)
    best_state = deepcopy(model.state_dict())
    best_epoch = 0

    print(f"validation before training: {best_val_loss:.4f}")

    for epoch in range(1, num_epochs + 1):
        model.train()

        for activity_batch, speed_batch in train_loader:
            targets = (speed_batch - speed_mean) / speed_std

            optimizer.zero_grad()
            predictions = model(activity_batch, positions_tensor, token_ids)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

        val_loss = evaluate(model, val_loader)
        print(f"epoch {epoch}: validation loss = {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch

    model.load_state_dict(best_state)
    print(f"best epoch: {best_epoch}, validation loss: {best_val_loss:.4f}")

    return model, best_val_loss, best_epoch

results = []
for model_class in (LinearSpeedDecoder, RunningSpeedDecoder, NoPositionSpeedDecoder):
    for seed in (0, 1, 2):
        print(f"\n{model_class.__name__}, seed={seed}")
        model, best_val_loss, best_epoch = train_model(model_class, seed)
        results.append((model_class.__name__, seed, best_val_loss, best_epoch))

