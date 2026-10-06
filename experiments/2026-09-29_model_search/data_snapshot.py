from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

path = Path(__file__).parent / "data" / "stringer_spontaneous.npy"
recording = np.load(path, allow_pickle=True).item()

activity = recording["sresp"]
positions = recording["xyz"].T
speed = recording["run"][:, 0]

rng = np.random.default_rng(0)
neuron_ids = rng.choice(activity.shape[0], size=128, replace=False)

activity = activity[neuron_ids]
positions = positions[neuron_ids]

#training/validation/testing split w/ gaps
n_time = activity.shape[1]
train_end = int(0.6 * n_time)
val_end = int(0.8 * n_time)
gap = 50

train_activity = activity[:, :train_end - gap]
train_speed = speed[:train_end - gap]

val_activity = activity[:, train_end + gap:val_end - gap]
val_speed = speed[train_end + gap:val_end - gap]

test_activity = activity[:, val_end + gap:]
test_speed = speed[val_end + gap:]

#ACTIVITY NORMALIZATION
activity_mean = train_activity.mean(axis=1, keepdims=True)
activity_std = train_activity.std(axis=1, keepdims=True)
activity_std = np.maximum(activity_std, 1e-6) #no div/0

train_activity = (train_activity - activity_mean) / activity_std
val_activity = (val_activity - activity_mean) / activity_std
test_activity = (test_activity - activity_mean) / activity_std

def make_examples(neural_activity, running_speed, window=8):
    inputs = []
    targets = []

    for target in range(window - 1, neural_activity.shape[1]):
        inputs.append(neural_activity[:, target - window + 1: target + 1])
        targets.append(running_speed[target])

    return np.stack(inputs), np.array(targets)

X_train, y_train = make_examples(train_activity, train_speed) #(4153, 128, 8), (4153,)
X_val, y_val = make_examples(val_activity, val_speed)
X_test, y_test = make_examples(test_activity, test_speed)

#pair inputs + targets, delivered in batches
def make_loader(inputs, targets, shuffle=False):
    dataset = TensorDataset(
        torch.tensor(inputs, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32),
    )

    return DataLoader(
        dataset,
        batch_size=32,
        shuffle=shuffle,
    generator=torch.Generator().manual_seed(0),
    )

train_loader = make_loader(X_train, y_train, shuffle=True)
val_loader = make_loader(X_val, y_val)
test_loader = make_loader(X_test, y_test)

#POSITION NORMALIZATION
position_mean = positions.mean(axis=0, keepdims=True)
position_std = positions.std(axis=0, keepdims=True)
position_std = np.maximum(position_std, 1e-6)

normalized_positions = (positions - position_mean) / position_std
positions_tensor = torch.tensor(normalized_positions, dtype=torch.float32)