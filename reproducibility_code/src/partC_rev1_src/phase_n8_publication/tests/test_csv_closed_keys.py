import pytest
from partC_rev1_src.phase_n8_publication.authority_csv_check import compute


def test_closed_reference_retains_original_key_when_its_flow_becomes_zero():
    identity=dict(suite_id='s03',reference_id='closed',graph_identity='masked',mapping_identity='same')
    ref=[dict(identity,abstract_link_id='closed-key',lambda_reference='0'),dict(identity,abstract_link_id='open',lambda_reference='3')]
    pred=[dict(identity,abstract_link_id=x,lambda_estimate=y,method_id='M1',state_id='fresh') for x,y in [('closed-key','0'),('open','2')]]
    keys=[dict(suite_id='s03',abstract_link_id='closed-key')]
    summary,_=compute(ref,pred,keys,reference_method_id='REFERENCE_CLOSED')
    key=next(r for r in summary if r['method_id']=='M1' and r['population']=='KEY_LINKS')
    assert key['link_count']==1 and key['wmape_status']=='UNDEFINED_ZERO_REFERENCE_TOTAL'
    with pytest.raises(ValueError,match='nonpositive reference'):compute(ref,pred,keys)
