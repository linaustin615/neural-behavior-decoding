from pathlib import Path
import json
import numpy as np
import torch
import architecture_search as q
from architecture_model import ArchitectureCandidate

q.initialize()
ROOT = Path(__file__).resolve().parent
checkpoint = torch.load(ROOT / 'runs/latent32_n2048_p606_s11_real.pt', weights_only=False)
model = ArchitectureCandidate(checkpoint['config'])
model.load_state_dict(checkpoint['state_dict'])
model.eval()
x, _ = q.get_data(2048, 606, 'val')
with torch.no_grad():
    tokens = model.base.encoder.tokenizer(x[:1], checkpoint['positions'], torch.arange(2048))
    _, weights = model.readin(model.latents, tokens, tokens, need_weights=True, average_attn_weights=False)
a = weights[0, 0].numpy().astype(np.float64)
row_sum_error = float(np.max(np.abs(a.sum(1) - 1)))
np.testing.assert_allclose(a.sum(1), 1, atol=1e-6, rtol=0)
cosines = [float(np.dot(a[i], a[j]) / (np.linalg.norm(a[i]) * np.linalg.norm(a[j]))) for i in range(32) for j in range(i)]
sv = np.linalg.svd(a, compute_uv=False)
rank = float(np.sum(sv ** 2) ** 2 / np.sum(sv ** 4))
gram = np.einsum('ik,jk->ij', a, a, optimize=False)
assert np.isfinite(gram).all()
np.testing.assert_allclose(rank, np.trace(gram) ** 2 / np.sum(gram ** 2), rtol=1e-10)
result = dict(task='latent32,pool606,seed11; first validation example,first head', attention_shape=list(weights.shape), softmax_max_row_sum_error=row_sum_error, softmax_sum_tolerance=1e-6, numpy_pairwise_cosine=float(np.mean(cosines)), numpy_svd_participation_rank=rank, gram_and_svd_agree=True, tolerance_note='Initial row-sum check used total tolerance3e-7; observed4.51e-7 error in float32 softmax. Use1e-6 for this numerical check; no model, metric, or experimental settings changed.')
(ROOT / 'summary_diversity_independent_check.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
