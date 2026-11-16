from partC_rev1_src.phase_n8.forecast import trajectory_estimate


def test_short_flat_and_disagreeing_fits_remain_unknown():
    assert trajectory_estimate([dict(iteration=1,RG_SPTT=.1)])['high_remaining_iterations'] is None
    flat=[dict(iteration=i,RG_SPTT=.1) for i in range(1,8)]
    assert trajectory_estimate(flat)['high_remaining_iterations'] is None
    # A strong geometric fall and a power-law extrapolation can disagree.
    falling=[dict(iteration=i,RG_SPTT=.9**i) for i in range(1,8)]
    assert trajectory_estimate(falling)['high_remaining_iterations'] is None
