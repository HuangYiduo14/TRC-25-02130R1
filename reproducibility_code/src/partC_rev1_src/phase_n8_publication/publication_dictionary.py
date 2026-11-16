"""Units and estimands for the native-coordinate publication CSVs."""
from partC_rev1_src.phase_n8_r4.release_data import field_meaning as prior_meaning


def field_meaning(col):
    exact={
        'lambda_reference':'Forward-mapped physical reference flow on this native abstract coordinate, annual kilotons.',
        'lambda_estimate':'Native abstract assignment flow on the same coordinate, annual kilotons; no physical-flow reconstruction.',
        'signed_bounded_error':'(estimate-reference)/max(abs(estimate),abs(reference)); exactly zero when both vanish; dimensionless in [-1,1].',
        'signed_relative_error':'(estimate-reference)/reference; defined only for strictly positive reference; dimensionless.',
        'absolute_relative_error':'abs(estimate-reference)/reference; defined only for strictly positive reference; dimensionless.',
        'backbone_wmape_contribution_percent':'100 times this coordinate absolute error divided by total complete-backbone reference flow.',
        'wmape_percent':'100*sum(abs(prediction-reference))/sum(reference) on the explicitly named complete population; percent.',
        'cost_wmape_percent':'100*sum(abs(predicted native marginal cost-true native marginal cost))/sum(true native marginal cost) across all admitted queries and coordinates in this region/cohort; percent.',
        'cost_wmape_query_percent':'The direct native marginal-cost WMAPE for one input vector; percent.',
        'cost_abs_error_sum':'Sum of absolute native marginal-cost prediction errors, seconds, across the named query/region/cohort population.',
        'true_cost_sum':'Sum of native true marginal costs, seconds, across exactly the same coordinates as the cost-error numerator.',
        'reference_backbone_cost':'Sum of reference native flow times native base+fourth-power marginal cost on the declared backbone population; annual kiloton-seconds.',
        'predicted_backbone_cost':'Sum of predicted native flow times the same native base+fourth-power marginal cost; annual kiloton-seconds.',
        'backbone_cost_error_percent':'100*abs(predicted backbone cost-reference backbone cost)/reference backbone cost; percent.',
        'redistribution_percent':'100*sum(abs(closed-open flow))/sum(open flow) on the declared population; response magnitude, not accuracy.',
        'online_seconds':'Complete supervised process wall time, seconds; exact initialization/reuse and nested-component boundaries are in timing_scope.',
        'recorded_attributable_setup_seconds':'Measured discarded discovery-epoch sum for the safety portfolio only; missing for other methods. It is not complete offline cost.',
        'is_key_link':'Whether this native abstract ID belongs to the pre-frozen 200-link subset; unchanged for closed-reference comparisons.',
        'flow_open':'Native abstract open-state flow, annual kilotons.',
        'flow_closed':'Native abstract closed-state flow, annual kilotons.',
        'flow_delta':'Closed minus open native abstract flow, annual kilotons.',
        'unavailable':'True iff the exact typed represented-I95 overlay disables this native abstract coordinate.',
        'selected_updates':'Fixed-VALID-selected neural checkpoint update; zero may select initialization after 4,000 updates were executed.',
        'executed_updates':'Actual number of optimizer updates executed, distinct from the selected checkpoint update.',
        'n_train':'Distinct TRAIN inputs available in this frozen nested subset; not necessarily all consumed by replacement sampling.',
        'source_TRAIN_count':'Training-row count in the frozen region-selection source record; distinct from a fraction-specific subset or optimizer exposure count.',
        'frozen_VALID_MARE':'Frozen source VALID mean absolute relative error used for region selection; dimensionless fraction, not later evaluation error.',
        'frozen_TRAIN_VALID_max_MARE':'Maximum frozen source TRAIN/VALID mean absolute relative error used for difficult-region selection; dimensionless fraction.',
        'selection_tie_sha256':'Stored deterministic hash used to resolve ties in the frozen regional selection.',
        'rank_within_stratum':'Preselected region rank within its declared stratum; established before the learning-study evaluation.',
        'label_collection_status':'Whether post-freeze operating/stress labels are reused unchanged or newly attempted, or the cohort is a reopened now-seen source audit.',
        'fit_reused':'True for unchanged previously completed fits; false for newly attempted missing-region fits.',
        'served_OD_membership_equal':'Exact equality of abstract and physical closed served-OD membership on the unchanged original aggregated OD order; not merely equality of totals.',
        'crs':'Geometry coordinate reference system; EPSG:5070 projected metres.',
        'vertex_order':'Ordered vertex index within one feature and part; only consecutive vertices of that same part are joined.',
        'part_id':'Multipart geometry identifier within a feature; separate parts are never connected.',
        'feature_id':'Stable geometry feature identifier; native endpoint chords and physical owned-road polylines remain distinct layers.'}
    if col in exact:return exact[col]
    if col in ('open_flow','closed_flow','signed_change','signed_error','absolute_error','reference_flow','reference_flow_sum','absolute_error_sum','open_flow_sum','absolute_change_sum','net_traversal_change'):
        return 'Native abstract flow or explicitly named sum/change, annual kilotons; population/role in the same record.'
    if col.endswith('_wmape') or col in ('backbone_cost_error','key_wmape'):
        return 'Named native-coordinate error expressed as a percentage; exact numerator/denominator supplied in the companion row.'
    if col in ('x0','x1','y0','y1','tail_x','tail_y','head_x','head_y'):
        return 'Native endpoint projected coordinate, EPSG:5070, metres; chords are not physical routes.'
    return prior_meaning(col)
