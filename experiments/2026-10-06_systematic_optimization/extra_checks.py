"""Check ridge mathematics and validation isolation without candidate fitting."""
import ast
import warnings
import numpy as np
from common import ROOT, write
import data
import fit
import ridge

fit.initialize_worker()
warnings.filterwarnings('error', category=RuntimeWarning)
rng = np.random.default_rng(708)
x, y, xv = rng.normal(size=(13,3,4)), rng.normal(size=13), rng.normal(size=(7,3,4))
predictions, states, _ = ridge.fit_path(x,y,xv,[.0001,.1,10.])
for lam, p, state in zip([.0001,.1,10.],predictions,states):
    z = (x.reshape(13,-1)-state['center'])/state['scale']
    v = (xv.reshape(7,-1)-state['center'])/state['scale']
    gram = np.einsum('ni,nj->ij',z,z,optimize=False)
    rhs = np.einsum('ni,n->i',z,y-y.mean(),optimize=False)
    w = np.linalg.solve(gram+lam*13*np.eye(12),rhs)
    expected = np.einsum('ni,i->n',v,w,optimize=False)+y.mean()
    assert np.isfinite(expected).all()
    np.testing.assert_allclose(p,expected,rtol=1e-9,atol=1e-9)
    np.testing.assert_allclose(ridge.predict(xv,state),p,rtol=1e-9,atol=1e-9)
seq, labels = rng.normal(size=(350,512)), rng.normal(size=350)
a, ma = data.scaled(seq,labels,200,(seq[264:],labels[264:]),-2.)
b, mb = data.scaled(seq,labels,200,(seq[264:]*30,labels[264:]+100),-2.)
for key in ['train_seq','train_y','activity_mean','activity_std']:
    np.testing.assert_array_equal(a[key],b[key])
for key in ['speed_mean','speed_std','lower']:
    assert ma[key] == mb[key]
for path in ROOT.glob('*.py'):
    ast.parse(path.read_text())
write(ROOT/'ridge_selfcheck.json',dict(passed=True,dual_matches_independent_primal=True,penalties_checked=3,
    validation_changes_do_not_change_fit_scaling=True,source_syntax_valid=True,strict_runtime_warnings=True,
    prefreeze_diagnostic_note='Initial ad-hoc NumPy matmul cross-check emitted runtime warnings despite finite matching outputs. This reproducible independent check uses explicit einsum, verifies finite outputs, and treats warnings as errors. Candidate training and Torch ridge math were unchanged.'),replace=True)
print('Strict ridge primal/dual and fold-scaling isolation checks passed',flush=True)
