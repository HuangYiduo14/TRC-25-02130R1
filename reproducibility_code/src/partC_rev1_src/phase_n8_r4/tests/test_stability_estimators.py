"""Ownership labels are categorical; relabeling must not change stability."""
import pytest
from partC_rev1_src.phase_n8_r4.stability import information


def test_exact_relabeling():
    result,table=information(['a','a','b','b'],['z','z','q','q'])
    assert result['ARI']==1
    assert result['NMI_arithmetic']==pytest.approx(1)
    assert table.physical_observations.sum()==4


def test_independent_balanced_partitions():
    result,table=information(['a','a','b','b'],['x','y','x','y'])
    assert result['ARI']==pytest.approx(-.5)
    assert result['NMI_arithmetic']==0.


def test_single_group_convention():
    result,_=information(['a']*4,['different']*4)
    assert result['ARI']==1. and result['NMI_arithmetic']==1.
