"""Synthetic deadline/C01/packaging integration; no campaign arrays or rendering."""
import copy
import json
from pathlib import Path

import pytest

from partC_rev1_src.pacnhn_v1 import continuation as c
from partC_rev1_src.pacnhn_v1 import handoff as h
from partC_rev1_src.pacnhn_v1 import publication as p
from partC_rev1_src.pacnhn_v1.tests.test_pacnhn_handoff import make_case, put, refresh

AUTH = 1790531577
HARD = 1790617977
NOW = AUTH+1100


def fixture(tmp_path, monkeypatch):
    case = make_case(tmp_path)
    repo, out, manifest, plan, audit = case
    contract = repo/h.OWNED[1]/"APPROVED_PLAN.md"
    original = dict(phase_id=h.PHASE, run_id="synthetic", original_start_epoch=h.START,
        hard_deadline_epoch=h.HARD_DEADLINE, output_root=str(out), contract_path=str(contract),
        contract_sha256=h.digest(contract), tasks=["publication", "handoff"], input_versions={"synthetic": "only"},
        expected_outputs=["synthetic"], acceptance_tests=["synthetic"],
        resources=dict(tree_rss_bytes=2**40, minimum_available_bytes=0))
    original_path=out/"reports/original_config.json";put(original_path,original)
    registry=out/"training/REGISTRY_FREEZE.json";put(registry,{"synthetic": True})
    result=out/"training/RESULT.json";put(result,dict(status="STAGE2_FINISHED_WITH_EXPLICIT_COVERAGE",
        registry=str(registry),registry_sha256=h.digest(registry)))
    common=dict(phase_id=h.PHASE,run_id="synthetic",authorization_epoch=AUTH,additional_seconds=86400,
        hard_deadline_epoch=HARD,original_start_epoch=h.START,original_hard_deadline_epoch=h.HARD_DEADLINE,
        stage3_max_seconds=43200,publication_max_seconds=14400,publication_reserve_seconds=14400)
    approval=dict(common,schema="pacnhn.continuation.approval.v1",decision="AUTHORIZE_TIME_EXTENSION_24H",
        authority_kind="explicit_user_message",user_statement="I hereby give additional 24 hours (from now). You can keep working.",
        authorization_turn_id="01a0e3ff-a4af-79d1-b5be-72c4affce7c3",original_config_path=str(original_path),
        original_config_sha256=h.digest(original_path),original_contract_sha256=h.digest(contract),
        frozen_registry_sha256=h.digest(registry),allowed_actions=sorted(c._ALLOWED_ACTIONS),
        retraining_authorized=False,scientific_settings_change_authorized=False,next_campaign_authorized=False,
        automatic_technical_gate_continuation=True)
    approval_path=out/"reports/approval.json";put(approval_path,approval)
    clock=dict(common,schema="pacnhn.continuation.clock.v1",continuation_start_epoch=AUTH,
        science_deadline_epoch=HARD-14400,clock_reset_allowed=False,
        approval_path=str(approval_path),approval_sha256=h.digest(approval_path))
    clock_path=out/"reports/continuation_clock.json";put(clock_path,clock)
    paths=dict(approval=approval_path,clock=clock_path,original_config=original_path,
        training_result=result,registry=registry)
    spec={"schema":"pacnhn.continuation.config.v1"}
    for key,path in paths.items():spec[key+"_path"]=str(path);spec[key+"_sha256"]=h.digest(path)
    monkeypatch.setattr(c,"_EXPECTED_PINS",{key:spec[key] for key in c._EXPECTED_PINS})
    config=dict(original,continuation=spec);config_path=out/"reports/config.json";put(config_path,config)
    refresh(plan,out,"CONFIG");plan["campaign_config_sha256"]=h.digest(config_path)
    identity=dict(phase_id=h.PHASE,run_id="synthetic",campaign_config_sha256=h.digest(config_path),
        original_contract_sha256=h.digest(contract),continuation_approval_sha256=h.digest(approval_path),
        continuation_clock_sha256=h.digest(clock_path))
    stage=dict(identity,schema="PACNHN_STAGE4_CLOCK_V1",stage_start_epoch=AUTH+1000,
        stage_deadline_epoch=AUTH+15400)
    stage_path=out/"STAGE4_CLOCK.json";put(stage_path,stage)
    amendment=dict(identity,schema="PACNHN_C01_CLOCK_AMENDMENT_V1",requirement_id="C01",
        original_start_epoch=h.START,original_hard_deadline_epoch=h.HARD_DEADLINE,
        original_52h_requirement_met=False,disclosure="Original52h delivery incomplete; separate authority.")
    amendment_path=out/"reports/clock_amendment.json";put(amendment_path,amendment)
    plan.update(campaign_config_path=str(config_path),deadline_epoch=stage["stage_deadline_epoch"],
        stage4_clock=dict(path=str(stage_path),sha256=h.digest(stage_path)),
        clock_amendment_member="governing_docs/clock_amendment.json")
    plan["approved_output_prefixes"].extend(["training/RESULT.json","STAGE4_CLOCK.json"])
    def include(path,member,role):
        row=dict(source_root="OUT",source_relative_path=path.relative_to(out).as_posix(),archive_path=member,
            sha256=h.digest(path),size_bytes=path.stat().st_size,role=role)
        plan["includes"].append(row);return row
    for name in ("approval","clock","original_config","training_result"):
        include(paths[name],"governing_docs/"+name+".json","CONTINUATION_AUTHORITY")
    include(stage_path,"governing_docs/stage4.json","CONTINUATION_CLOCK")
    amendment_row=include(amendment_path,plan["clock_amendment_member"],"CLOCK_AMENDMENT")
    c01=next(row for row in audit["requirements"] if row["id"]=="C01")["final_audit"]
    c01.update(audited_epoch=NOW,disposition_reason="Original52h incomplete, separately authorized continuation.")
    c01["evidence_bindings"].append(dict(path=amendment_row["archive_path"],sha256=amendment_row["sha256"],
        bytes=amendment_row["size_bytes"],role="CLOCK_AMENDMENT"))
    put(out/"reports/audit.json",audit);refresh(plan,out,"FINAL_AUDIT")
    pub=dict(phase_id=h.PHASE,step_id="publication",run_id="synthetic",campaign_config=str(config_path),
        manuscript_evidence={key: None for key in ("training_result", "training_clock", "finalization_clock",
            "finalization_forecast", "test_summary", "test_tail_report", "m02_result", "m02_seal",
            "t08_result", "t08_intent", "t08_manifest", "t08_seal")},
        inputs={"synthetic":"not loaded"},output_root=str(out/"publication"),tasks=["synthetic"],
        expected_outputs=["synthetic"],acceptance_tests=["synthetic"],resource_budget={"synthetic":True},
        deadline_epoch=stage["stage_deadline_epoch"],stage4_clock=plan["stage4_clock"],
        clock_amendment=dict(path=str(amendment_path),sha256=h.digest(amendment_path)))
    pub_path=out/"publication_config.json";put(pub_path,pub)
    return dict(case=case,config=config,config_path=config_path,stage=stage,stage_path=stage_path,
        amendment=amendment,amendment_path=amendment_path,approval_path=approval_path,
        pub=pub,pub_path=pub_path)


def window(f, now=NOW):
    return h.publication_window(f["config"],campaign_config_sha256=h.digest(f["config_path"]),
        stage4_clock=f["pub"]["stage4_clock"],clock_amendment=f["pub"]["clock_amendment"],
        deadline_epoch=f["pub"]["deadline_epoch"],now=now)


def execute(f, now=NOW):
    repo,out,manifest,plan,audit=f["case"];put(manifest,plan)
    return h.create_handoff(manifest,repo_root=repo,output_root=out,guard=lambda:None,now=lambda:now)


def test_authenticated_extension_has_separate_four_hour_clock_and_disclosure(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch);w=window(f)
    assert w["deadline_epoch"]==AUTH+15400>h.HARD_DEADLINE
    assert w["original_52h_requirement_met"] is False
    assert "not met" in w["disclosure"] and "24-hour" in w["disclosure"]
    assert f["config"]["hard_deadline_epoch"]==h.HARD_DEADLINE


def test_publication_reader_uses_actual_extension_and_bound_stage4(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch);monkeypatch.setattr(p.time,"time",lambda:NOW)
    from partC_rev1_src.pacnhn_v1 import gates
    seen=[]
    monkeypatch.setattr(gates,"require_component",lambda ctx,component:seen.append(component))
    cfg,ctx,root=p.read_contract(f["pub_path"])
    assert seen==["publication"]
    assert ctx.publication_continuation_window["original_52h_requirement_met"] is False
    assert root==f["case"][1]/"publication" and not root.exists()
    assert str(f["stage_path"]) in ctx.bindings


def test_callerless_packager_reopens_extension_with_all51_rows(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch);receipt=execute(f)
    assert receipt["verification"]["member_sha256"]=="PASS"
    assert receipt["populated_requirement_count"]==51
    assert receipt["continuation_window"]["original_52h_requirement_met"] is False
    assert receipt["external_acceptance"]=="NOT_RECORDED"


@pytest.mark.parametrize("field,value",[("stage_start_epoch",NOW+1),("stage_deadline_epoch",HARD+1),
    ("stage_start_epoch",AUTH-1),("stage_deadline_epoch",AUTH+15401),("stage_start_epoch",float("nan")),
    ("campaign_config_sha256","0"*64),("continuation_clock_sha256","0"*64)])
def test_invalid_rebound_stage4_metadata_rejected(tmp_path,monkeypatch,field,value):
    f=fixture(tmp_path,monkeypatch);f["stage"][field]=value;put(f["stage_path"],f["stage"])
    f["pub"]["stage4_clock"]["sha256"]=h.digest(f["stage_path"])
    with pytest.raises((ValueError,TimeoutError)):window(f)


@pytest.mark.parametrize("field,value",[("original_52h_requirement_met",True),("original_hard_deadline_epoch",HARD),
    ("requirement_id","C02"),("disclosure",""),("continuation_approval_sha256","0"*64)])
def test_c01_cannot_hide_original_deadline_or_change_authority(tmp_path,monkeypatch,field,value):
    f=fixture(tmp_path,monkeypatch);f["amendment"][field]=value;put(f["amendment_path"],f["amendment"])
    f["pub"]["clock_amendment"]["sha256"]=h.digest(f["amendment_path"])
    with pytest.raises(h.HandoffRejected):window(f)


def test_bound_clock_byte_change_is_rejected_without_rebinding(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch);put(f["stage_path"],dict(f["stage"],changed=True))
    with pytest.raises(h.HandoffRejected,match="bytes changed"):window(f)


@pytest.mark.parametrize("now",[AUTH+15400,HARD])
def test_expired_stage4_or_total_window_rejected(tmp_path,monkeypatch,now):
    with pytest.raises((h.HandoffRejected,TimeoutError)):window(fixture(tmp_path,monkeypatch),now=now)


def test_stage4_late_start_is_clamped_to_total_deadline(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch);f["stage"].update(stage_start_epoch=HARD-100,stage_deadline_epoch=HARD)
    put(f["stage_path"],f["stage"]);f["pub"]["stage4_clock"]["sha256"]=h.digest(f["stage_path"])
    f["pub"]["deadline_epoch"]=HARD
    assert window(f,now=HARD-50)["stage_deadline_epoch"]==HARD


@pytest.mark.parametrize("status",["PASS","NOT_APPLICABLE_WITH_REASON"])
def test_final_C01_rejects_unqualified_pass_or_not_applicable(tmp_path,monkeypatch,status):
    f=fixture(tmp_path,monkeypatch);case=f["case"]
    next(row for row in case[4]["requirements"] if row["id"]=="C01")["final_audit"]["status"]=status
    put(case[1]/"reports/audit.json",case[4]);refresh(case[3],case[1],"FINAL_AUDIT")
    with pytest.raises(h.HandoffRejected) as caught:execute(f)
    assert caught.value.code=="C01_DISCLOSURE"


def test_final_C01_requires_exact_supplement_citation(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch);case=f["case"]
    next(row for row in case[4]["requirements"] if row["id"]=="C01")["final_audit"]["evidence_bindings"].pop()
    put(case[1]/"reports/audit.json",case[4]);refresh(case[3],case[1],"FINAL_AUDIT")
    with pytest.raises(h.HandoffRejected) as caught:execute(f)
    assert caught.value.code=="C01_DISCLOSURE"


def test_full_authority_evidence_cannot_be_omitted_from_archive(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch);plan=f["case"][3]
    plan["includes"]=[row for row in plan["includes"] if row["archive_path"]!="governing_docs/approval.json"]
    with pytest.raises(h.HandoffRejected) as caught:execute(f)
    assert caught.value.code=="MISSING_CONTINUATION_EVIDENCE"


def test_extended_callerless_plan_cannot_omit_config_authentication(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch);del f["case"][3]["campaign_config_path"]
    with pytest.raises(h.HandoffRejected) as caught:execute(f)
    assert caught.value.code=="CLOCK_CHANGED"


def test_authority_file_mutation_not_accepted_even_with_permissive_caller_guard(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch);put(f["approval_path"],dict(h.read_json(f["approval_path"]),extra=True))
    with pytest.raises(ValueError,match="changed"):execute(f)
    assert not (f["case"][1]/"handoff_attempts/synthetic_v1/FINAL_SEAL_RECEIPT.json").exists()


def test_legacy_publication_retains_old_cap_without_extension(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch);config=copy.deepcopy(f["config"]);config.pop("continuation")
    put(f["config_path"],config);f["pub"]["deadline_epoch"]=h.START+200;put(f["pub_path"],f["pub"])
    monkeypatch.setattr(p.time,"time",lambda:h.START+100)
    _,ctx,_=p.read_contract(f["pub_path"]);assert ctx.publication_continuation_window is None
    monkeypatch.setattr(p.time,"time",lambda:NOW)
    with pytest.raises(TimeoutError):p.read_contract(f["pub_path"])


def test_alternate_stage4_path_cannot_start_another_clock(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch)
    alternate=f["case"][1]/"another_stage4.json";put(alternate,f["stage"])
    f["pub"]["stage4_clock"]=dict(path=str(alternate),sha256=h.digest(alternate))
    with pytest.raises(h.HandoffRejected,match="canonical"):window(f)


def test_generated_numerical_prose_discloses_authorized_original_clock_shortfall(tmp_path,monkeypatch):
    import pandas as pd
    f=fixture(tmp_path,monkeypatch);note=window(f)["disclosure"]
    data={name:pd.DataFrame(columns=cols) for name,cols in p.SCHEMAS.items()}
    sections=p.manuscript_sections(data,clock_disclosure=note)
    timing=next(paragraphs for title,paragraphs in sections if title.startswith("Timing"))
    assert any("original 52-hour delivery window was not met" in paragraph for paragraph in timing)
    assert any("separately approved 24-hour extension" in paragraph for paragraph in timing)
    assert not any(note in paragraph or "C01" in paragraph for paragraph in timing)
    assert not any("original 52-hour campaign clock includes all stages" in paragraph for paragraph in timing)
    legacy=p.manuscript_sections(data)
    assert any("original 52-hour campaign clock includes all stages" in paragraph
               for title,paragraphs in legacy for paragraph in paragraphs)



def learning_fixture():
    import pandas as pd
    rows=[]
    for region, error, actual, percent in [(0, 1., 1., 100.), (1, 0., 99., 0.)]:
        row={column: "" for column in p.SCHEMAS["learning"]}
        row.update(suite_id="s03", region_index=region, family="POWER4", status="AVAILABLE", target_met=True,
            train_rows=2, parameters=2, neural_amplitude="", TRAIN_WMAPE_percent=.1, VALID_WMAPE_percent=.2,
            TEST_WMAPE_percent=percent, test_opened=True, TEST_absolute_error_sum=error,
            TEST_true_cost_sum=actual, TEST_coordinates=6, TEST_inputs=3)
        rows.append(row)
    return pd.DataFrame(rows)


def test_pooled_test_uses_ratio_of_explicit_sums_not_median_or_mean():
    frame=learning_fixture(); result=p.pooled_test_summary(frame)
    assert result["wmape_percent"]==1 and frame.TEST_WMAPE_percent.median()==50
    assert result["coordinates"]==12 and result["inputs"]==6 and result["contributing_regions"]==2
    note=p.pooled_test_narrative(frame)
    assert "1% pooled versus 50% unweighted regional median" in note
    assert "region-input observations" in note and "regional TEST maximum is 100%" in note


def test_pooled_test_excludes_unopened_unavailable_and_incomplete_evidence():
    import pandas as pd
    frame=learning_fixture()
    unopened=frame.iloc[0].copy(); unopened["test_opened"]=False; unopened["TEST_absolute_error_sum"]=1e9
    missing=frame.iloc[0].copy(); missing["TEST_true_cost_sum"]=""
    unavailable=frame.iloc[0].copy(); unavailable["status"]="UNAVAILABLE_STALE_CURRENT_DATASET"; unavailable["TEST_absolute_error_sum"]=1e9
    frame=pd.concat([frame,pd.DataFrame([unopened,missing,unavailable])],ignore_index=True)
    result=p.pooled_test_summary(frame)
    assert result["wmape_percent"]==1 and result["contributing_regions"]==2
    assert result["excluded_regions"]==3 and result["unavailable_regions"]==1
    assert "3 excluded, 1 unavailable model records" in p.pooled_test_narrative(frame)


@pytest.mark.parametrize("missing",p.TEST_AGGREGATE_FIELDS+["test_opened"])
def test_pooled_test_legacy_or_missing_denominator_is_explicitly_unavailable(missing):
    result=p.pooled_test_summary(learning_fixture().drop(columns=missing))
    assert result["wmape_percent"] is None and result["contributing_regions"]==0
    assert "N/A" in p.pooled_test_narrative(learning_fixture().drop(columns=missing))


@pytest.mark.parametrize("field,value",[("TEST_absolute_error_sum",-1),("TEST_absolute_error_sum","bad"),("TEST_true_cost_sum",float("inf")),
    ("TEST_coordinates",1.5),("TEST_inputs",-1),("TEST_inputs",7),("TEST_coordinates",0)])
def test_pooled_test_rejects_invalid_supplied_aggregates(field,value):
    frame=learning_fixture();frame[field]=frame[field].astype(object);frame.loc[0,field]=value
    with pytest.raises(ValueError):p.pooled_test_summary(frame)


def test_pooled_test_preserves_measured_zero_and_zero_truth_is_undefined():
    frame=learning_fixture();frame["TEST_absolute_error_sum"]=0.
    result=p.pooled_test_summary(frame);assert result["wmape_percent"]==0 and result["contributing_regions"]==2
    frame["TEST_true_cost_sum"]=0.
    result=p.pooled_test_summary(frame)
    assert result["wmape_percent"] is None and result["status"]=="UNDEFINED_ZERO_TRUE_COST"
    assert result["contributing_regions"]==2 and result["true_cost_sum"]==0


def test_table3_retains_independent_aggregate_observations_and_both_statistics(tmp_path,monkeypatch):
    import pandas as pd
    data={name:pd.DataFrame(columns=cols) for name,cols in p.SCHEMAS.items()}
    data["learning"]=learning_fixture(); seen={}
    def capture(root,asset_id,frames,records,**kwargs):seen[asset_id]=(frames,kwargs)
    monkeypatch.setattr(p,"asset_folder",capture)
    p.build_tables(tmp_path,data,[])
    frames,kwargs=seen["table3"]; row=kwargs["table"].iloc[0]
    assert row["TEST"]=="50 (2)" and row["pooled_TEST"]=="1 (2/2)"
    pd.testing.assert_frame_equal(frames["observations_learning.csv"],data["learning"])
    assert frames["pooled_test_summary.csv"].iloc[0].true_cost_sum==100
    sections=p.manuscript_sections(data)
    assert any("1% pooled versus 50%" in paragraph for _,paragraphs in sections for paragraph in paragraphs)


def test_table3_legacy_sums_do_not_create_a_pooled_measurement(tmp_path,monkeypatch):
    import pandas as pd
    data={name:pd.DataFrame(columns=cols) for name,cols in p.SCHEMAS.items()}
    data["learning"]=learning_fixture().drop(columns=p.TEST_AGGREGATE_FIELDS);seen={}
    def capture(root,asset_id,frames,records,**kwargs):seen[asset_id]=kwargs
    monkeypatch.setattr(p,"asset_folder",capture);p.build_tables(tmp_path,data,[])
    assert seen["table3"]["table"].iloc[0].pooled_TEST=="N/A (0/2)"



def test_zero_truth_region_error_remains_in_pooled_numerator():
    frame=learning_fixture();frame.loc[0,"TEST_true_cost_sum"]=0
    result=p.pooled_test_summary(frame)
    assert result["absolute_error_sum"]==1 and result["true_cost_sum"]==99
    assert result["wmape_percent"]==pytest.approx(100/99) and result["contributing_regions"]==2


def test_analysis_csv_ingestion_preserves_all_explicit_test_sums(tmp_path):
    import pandas as pd
    data={name:pd.DataFrame(columns=cols) for name,cols in p.SCHEMAS.items()}
    data["learning"]=learning_fixture()
    data["network"]=pd.DataFrame([dict(suite_id="s03",resolution=1,nodes=2,edges=87583,
        backbone_links=87582,regional_links=1,regions=2,active_regions=2,active_pairs=2,
        served_rows_open=1,served_rows_closed=1)])
    data["keys"]=pd.DataFrame(dict(link_id=[str(i) for i in range(200)]))
    cfg=dict(deadline_epoch=NOW+100,inputs={})
    for name,frame in data.items():
        path=tmp_path/(name+".csv");frame.to_csv(path,index=False)
        cfg["inputs"][name]=dict(path=str(path),sha256=h.digest(path))
    class FakeContext:
        def guard(self,deadline):pass
        def bind(self,path,role,expected):
            assert h.digest(path)==expected
            return expected
    loaded,records=p.load_inputs(cfg,FakeContext())
    for column in p.TEST_AGGREGATE_FIELDS:
        assert column in loaded["learning"]
        assert list(loaded["learning"][column])==list(data["learning"][column])
    assert p.pooled_test_summary(loaded["learning"])["wmape_percent"]==1
    record=next(r for r in records if r["dataset"]=="learning")
    assert set(p.TEST_AGGREGATE_FIELDS).issubset(record["columns"])
