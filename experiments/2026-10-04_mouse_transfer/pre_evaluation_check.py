"""Bind preparation metadata and all selected choices before later scoring."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT=Path(__file__).resolve().parent


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    if sys.argv[1:]==['lock']:
        assert not (ROOT/'results.json').exists()
        assert not (ROOT/'pre_evaluation_lock.json').exists()
        prepared=json.loads((ROOT/'prepared.json').read_text())
        neural=json.loads((ROOT/'selection_lock.json').read_text())
        assert len(neural['records'])==72 and neural['selection_scores_checked']==3000
        checked=0
        for row in prepared['records']:
            with np.load(ROOT/row['mouse']/'ridge.npz') as z:
                errors=[]
                for p in z['selection_predictions']:
                    error=sum((max(float(v),row['lower'])-float(y))**2 for v,y in zip(p,z['target']))/len(p)
                    errors.append(error)
                    checked+=1
                np.testing.assert_allclose(errors,row['ridge_selection_mse'],rtol=1e-12,atol=1e-12)
                assert int(np.argmin(errors))==row['ridge_index']
        names=['protocol.json','prepared.json','selfcheck.json','prefix_audit.json','selection_lock.json','pre_evaluation_check.py']
        value=dict(locked_utc=datetime.now(timezone.utc).isoformat(),passed=True,
                   independent_ridge_selection_scores=checked,ridge_argmins_checked=4,later_scored=False,
                   hashes={name:digest(ROOT/name) for name in names})
        (ROOT/'pre_evaluation_lock.json').write_text(json.dumps(value,indent=2)+'\n')
        print('Locked preparation and neural choices;20 ridge selection scores independently checked')
    elif sys.argv[1:]==['verify']:
        value=json.loads((ROOT/'pre_evaluation_lock.json').read_text())
        for name,h in value['hashes'].items():
            assert digest(ROOT/name)==h,name
        print('Pre-evaluation metadata and choices remain unchanged')
    else:
        raise SystemExit('usage: pre_evaluation_check.py lock|verify')


if __name__=='__main__':
    main()
