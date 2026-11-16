import numpy as np
import pytest
from partC_rev1_src.phase_n8_publication.metrics import paired_reachability


def test_equal_demand_totals_do_not_imply_equal_served_od():
    r=paired_reachability([True,False,True],[False,True,True],[2,2,7],[3,4,1])
    assert not r['same_served_OD_membership']
    rows={x['population']:x for x in r['rows']}
    assert rows['ABSTRACT_SERVED_TOTAL']['demand_annual_ktons']==rows['PHYSICAL_SERVED_TOTAL']['demand_annual_ktons']==9
    assert rows['BOTH_REACHABLE']['demand_annual_ktons']==7
    assert rows['ABSTRACT_ONLY']['original_rows']==3
    assert rows['PHYSICAL_ONLY']['original_rows']==4


def test_identical_membership_keeps_unreachable_rows_in_ledger():
    r=paired_reachability([True,False],[True,False],[3,5],[2,7])
    assert r['same_served_OD_membership']
    assert next(x for x in r['rows'] if x['population']=='NEITHER_REACHABLE')['demand_annual_ktons']==5
    with pytest.raises(ValueError,match='dimension'):
        paired_reachability([True],[True,False],[3,5],[1,1])
