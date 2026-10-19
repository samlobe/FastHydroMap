"""Selected shared model invariants, optional embeddings and provenance."""
from pathlib import Path
import copy,json,hashlib
import numpy as np
import pytest
import torch
from FastHydroMap.v2.model import SharedWaterModel
from FastHydroMap.v2 import Predictor
from FastHydroMap.cli import main

DATA=Path(__file__).parent/'data';PDB=DATA/'v2_chain.pdb'

def synthetic():
    torch.manual_seed(19)
    return dict(node=torch.randn(8,35),point=torch.randn(8,24,11),atom=torch.randn(8,24,8,19),
        point_mask=torch.ones(8,24,dtype=torch.bool),edge_index=torch.tensor([[0,1,2,3],[1,0,3,2]]),edge=torch.randn(4,20))

def test_shared_shapes_locality_and_empty_probes():
    model=SharedWaterModel().eval();assert sum(p.numel() for p in model.parameters())==8740
    g=synthetic();g['point_mask'][0]=False
    with torch.no_grad():
        y,h=model(g,return_embeddings=True)
        assert y.shape==(8,4) and h.shape==(8,16) and torch.isfinite(y).all()
        torch.testing.assert_close(model.readout(h),y,atol=0,rtol=0)
        changed=copy.deepcopy(g)
        for k in ('node','point','atom'):changed[k][4:]=123
        torch.testing.assert_close(model(changed)[:4],y[:4],atol=0,rtol=0)
        changed=copy.deepcopy(g);changed['actual']=torch.full((8,4),999.);changed['waters']=torch.full((8,),999.)
        torch.testing.assert_close(model(changed),y,atol=0,rtol=0)
        changed=copy.deepcopy(g);changed['point'][0]=999.;changed['atom'][0]=999.
        torch.testing.assert_close(model(changed),y,atol=0,rtol=0)

def test_shared_export_invariance_and_regression():
    model=Predictor()
    if not model.shared:pytest.skip('Shared final candidate not staged yet')
    plain=model.predict(PDB);with_h=model.predict(PDB,embeddings=True)
    np.testing.assert_array_equal(plain[list(model.quantities)],with_h[list(model.quantities)])
    record=json.loads((DATA/'v2_shared_chain_expected.json').read_text());expected=pd_frame(record['records'])
    assert record['input_sha256']==hashlib.sha256(PDB.read_bytes()).hexdigest()
    assert record['manifest_sha256']==with_h.attrs['model_manifest_sha256']
    np.testing.assert_allclose(with_h[[f'embedding_{j}' for j in range(16)]],expected[[f'embedding_{j}' for j in range(16)]],atol=1e-5,rtol=0)
    assert with_h.attrs['embedding_seed']==17 and with_h.attrs['prediction_ensemble_seeds']==[17,29,43]

def pd_frame(records):
    import pandas as pd
    return pd.DataFrame(records)

def test_shared_cli_export(monkeypatch,tmp_path):
    if not Predictor().shared:pytest.skip('Shared final candidate not staged yet')
    root=tmp_path/'maps'
    monkeypatch.setattr('sys.argv',['fasthydromap','predict',str(PDB),'--model','v2','--quantity','all','--embeddings','-o',str(root)])
    main();archive=np.load(str(root)+'_embeddings.npz',allow_pickle=False)
    assert archive['embedding'].shape==(73,16) and int(archive['seed'])==17
    metadata=json.loads(Path(str(root)+'.json').read_text())
    assert metadata['embeddings_exported'] and metadata['embedding_metadata']['embedding_checkpoint_sha256']==str(archive['checkpoint_sha256'])
    with pytest.raises(SystemExit):main()
