"""Check every structural setting before freezing the optimization study."""
import importlib.util
from pathlib import Path
import tempfile
import time
import numpy as np
import torch
from common import ROOT, EXP, write
from models import make_model
from plan import baseline, screen_configs
import fit

fit.initialize_worker()
source = EXP/'2026-10-06_population_tokens'/'models.py'
spec = importlib.util.spec_from_file_location('optimization_population_parent', source)
parent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent)
checks = []
for config in screen_configs():
    h, p = config['history'], config['patch']
    torch.manual_seed(902)
    x = torch.randn(3, 512, h)
    before = x.clone()
    net = make_model(config, 101).eval()
    changed = x.clone()
    changed[:, :, h//2:] += 2
    with torch.no_grad():
        assert torch.equal(net(x, 0), torch.zeros(3))
        z, z2 = net.encode(x, 0), net.encode(changed, 0)
        torch.testing.assert_close(z[:, :h//(2*p)], z2[:, :h//(2*p)], rtol=0, atol=0)
        assert not torch.equal(z[:, -1], z2[:, -1])
        net.head[-1].weight.fill_(.01)
    loss = (net(x, 0)-torch.tensor([.5, -.2, 1.])).square().mean()
    loss.backward()
    assert all(torch.isfinite(v.grad).all() for v in net.parameters() if v.grad is not None)
    assert net.readin.grad[0].abs().sum() > 0 and net.readin.grad[1:].abs().sum() == 0
    torch.testing.assert_close(x, before, rtol=0, atol=0)
    clone = make_model(config, 9).eval()
    clone.load_state_dict(net.state_dict())
    with torch.no_grad():
        torch.testing.assert_close(net(x, 0), clone(x, 0), rtol=0, atol=0)
    checks.append(dict(config=config, parameters=sum(v.numel() for v in net.parameters()), causal_prefix=True, gradients=True, reload_exact=True))

equivalence = []
for family in ['attention', 'mlp']:
    for seed in [10, 11, 12]:
        old = parent.PopulationDecoder(family, seed, range(4)).eval()
        old.load_state_dict(torch.load(EXP/'2026-10-06_population_tokens'/f'{family}_s{seed}'/'selected.pt', weights_only=True))
        net = make_model(baseline(family), seed).eval()
        net.load_state_dict({k.replace('temporal.', 'layers.0.'): v for k, v in old.state_dict().items()})
        x = torch.randn(4, 512, 32)
        with torch.no_grad():
            torch.testing.assert_close(old(x, 0), net(x, 0), rtol=0, atol=0)
        equivalence.append(dict(family=family, seed=seed, trained_parent_exact=True))

a, b = [make_model(baseline(f), 101) for f in ['attention', 'mlp']]
common = [key for key in a.state_dict() if not key.startswith('layers.')]
assert all(torch.equal(a.state_dict()[key], b.state_dict()[key]) for key in common)
timings = []
for c in [baseline('attention'), dict(baseline('attention'), width=64, depth=3, history=64, patch=2), baseline('local_mlp')]:
    net = make_model(c, 101)
    x, y = torch.randn(64, 512, c['history']), torch.randn(64)
    opt = torch.optim.AdamW(net.parameters(), lr=.001)
    durations = []
    for step in range(7):
        start = time.monotonic()
        opt.zero_grad(set_to_none=True)
        (net(x, 0)-y).square().mean().backward()
        opt.step()
        if step >= 2:
            durations.append(time.monotonic()-start)
    timings.append(dict(config=c, median_step_seconds=float(np.median(durations))))

original_root, original_loader = fit.ROOT, fit.load_data
with tempfile.TemporaryDirectory(prefix='neuron_optimization_check_') as temporary:
    fit.ROOT = Path(temporary)
    def synthetic(split, history):
        torch.manual_seed(907)
        return {s: dict(x=torch.randn(10, 512, history), y=torch.randn(10), xv=torch.randn(4, 512, history),
            yv=np.array([.2, -.1, .4, 1.]), lower=-1., denominator=1.) for s in range(4)}
    fit.load_data = synthetic
    result = fit.fit(dict(config=baseline('attention'), split='synthetic', seed=101, epochs=2))
    assert np.isfinite(result['score'])
fit.ROOT, fit.load_data = original_root, original_loader
write(ROOT/'selfcheck.json', dict(passed=True, structural_configurations_checked=len(checks), checks=checks,
    trained_parent_equivalence=equivalence, common_initial_tensors=len(common), input_preservation=True,
    diagnostic_training_steps='Synthetic2epoch engine check and7step timing probes; discarded,not candidate fits',
    complete_training_engine_smoke_passed=True, timings=timings))
print('All108 structural settings,6 trained-parent matches and training-engine checks passed', flush=True)
print('Step timings', timings, flush=True)
