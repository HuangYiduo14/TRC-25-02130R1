from types import SimpleNamespace
from pathlib import Path
import pytest
from partC_rev1_src.pacnhn_v1.control import sha,write_json,read_json
from partC_rev1_src.pacnhn_v1.gates import topology_snapshot,require_component


@pytest.mark.parametrize('mutation',['graph_payload','od_payload','partition_payload','report_identity','config','extra_config','source'])
def test_preflight_rejects_changed_scientific_inputs_and_code(tmp_path,monkeypatch,mutation):
    import partC_rev1_src.pacnhn_v1.gates as gates
    root=tmp_path/'run';topology=root/'topology';topology.mkdir(parents=True)
    config=tmp_path/'campaign.json';extra=tmp_path/'training.json';source=tmp_path/'training.py'
    config.write_text('{}');extra.write_text('{}');source.write_text('VERSION=1')
    ctx=SimpleNamespace(root=root,path=config)
    report={k:k+'-frozen' for k in ('graph_identity','OD_identity','graph_array_identity','OD_array_identity','partitions_identity')}
    report['status']='PASS';write_json(topology/'RESULT.json',report)
    for name in ('common_graph.npz','common_od.npz','partitions.npz'):(topology/name).write_bytes(b'frozen payload')
    monkeypatch.setattr(gates,'component_snapshot',lambda *_:{str(source):sha(source)})
    write_json(root/'validation/PREFLIGHT_LATEST.json',dict(status='PASS',config_sha256=sha(config),
        component_snapshots={'training':gates.component_snapshot(ctx,'training')},
        topology_snapshot=topology_snapshot(ctx),extra_configs={str(extra.resolve()):sha(extra)}))
    require_component(ctx,'training',extra_config=extra)
    targets={'graph_payload':topology/'common_graph.npz','od_payload':topology/'common_od.npz',
             'partition_payload':topology/'partitions.npz','config':config,'extra_config':extra,'source':source}
    if mutation=='report_identity':
        report['OD_array_identity']='different';write_json(topology/'RESULT.json',report)
    else:targets[mutation].write_text('changed')
    with pytest.raises(ValueError):require_component(ctx,'training',extra_config=extra)
