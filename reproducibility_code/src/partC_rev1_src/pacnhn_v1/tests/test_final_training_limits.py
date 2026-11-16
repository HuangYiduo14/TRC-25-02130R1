"""Synthetic T08 checks only; no campaign arrays, labels, models or TEST access."""
import ast
import copy
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pytest

from partC_rev1_src.pacnhn_v1 import final_training_limits as a


def put(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    return binding(path)


def binding(path):
    return dict(path=str(path), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())


def npz(path, **values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.savez_compressed(stream, **values)
    return binding(path)


def data(u, tau):
    u = np.asarray(u, dtype=np.float64); tau = np.asarray(tau, dtype=np.float64)
    result = dict(u=u, tau=tau, c0=np.zeros(u.shape[1]),
                  records=[dict(input_identity=a.array_hash({"u": q})) for q in u])
    result["dataset_identity"] = a.array_hash({k: result[k] for k in ("u", "tau", "c0")})
    return result


def historical_screen():
    # Load ONLY this AST function: no training/model imports and no global data.
    path = Path(a.__file__).with_name("training.py")
    function = next(n for n in ast.parse(path.read_text(encoding="utf-8")).body
                    if isinstance(n, ast.FunctionDef) and n.name == "monotonic_conflicts")
    scope = {"np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), scope)
    return scope["monotonic_conflicts"]


@pytest.mark.parametrize("u,tau", [
    ([[0.], [1.], [2.], [3.]], [[4.], [2.], [5.], [1.]]),
    ([[0., 2.], [2., 0.], [4., 4.]], [[3., 1.], [1., 3.], [4., 4.]]),
    ([[0.], [1.]], [[0.], [0.]]),
    ([[1., 1.]], [[1., 2.]]),
])
def test_exact_historical_algorithm(u, tau):
    train = data(u, tau); scale = np.ones(train["u"].shape[1]); keys = np.arange(len(scale), dtype=np.uint64)
    expected = historical_screen()(train, scale, keys)
    expected = json.loads(json.dumps(expected, default=lambda v: v.tolist()))
    actual = a.monotonic_screen(train, scale, keys)
    assert actual == expected
    assert a.validate_screen(actual, train["dataset_identity"], len(train["u"])) is actual


def test_overlapping_violation_is_not_a_valid_lower_bound():
    train = data([[0.], [1.], [2.], [3.]], [[4.], [2.], [5.], [1.]])
    screen = a.monotonic_screen(train, np.ones(1), np.array([1], np.uint64))
    screen["violations"][1]["lower_row"] = 0
    with pytest.raises(ValueError, match="disjoint"):
        a.validate_screen(screen, train["dataset_identity"], 4)


@pytest.mark.parametrize("field,value", [("train_dataset_identity", "f"*64),
    ("fixed_label_L1_error_lower_bound", 1.), ("fixed_label_WMAPE_percent_lower_bound", 99.),
    ("screen", "EXHAUSTIVE"), ("ordered_comparisons", 100)])
def test_invalid_reused_receipt_rejected(field, value):
    train = data([[0.], [1.]], [[4.], [2.]])
    screen = a.monotonic_screen(train, np.ones(1), np.array([1], np.uint64)); screen[field] = value
    with pytest.raises(ValueError):
        a.validate_screen(screen, train["dataset_identity"], 2)


def test_reader_rejects_tamper_duplicate_nonfinite_and_size(tmp_path):
    b = put(tmp_path/"a.json", {"a": 1}); reader = a.Reader()
    reader.json(b)
    Path(b["path"]).write_text('{"a":2}', encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        reader.json(b)
    for raw in ('{"a":1,"a":2}', '{"a":NaN}'):
        p = tmp_path/"bad.json"; p.write_text(raw)
        with pytest.raises(ValueError):
            a.Reader().json(binding(p))
    with pytest.raises(ValueError, match="byte bound"):
        a.Reader().json(binding(p), maximum=1)


def label_fixture(tmp_path, *, key="s03:000936"):
    root = tmp_path/"labels"; root.mkdir(parents=True)
    graph = dict(tail=np.array([0], np.int64), head=np.array([1], np.int64),
        node_ids=np.array(["a", "b"]), edge_ids=np.array(["edge"]), base=np.array([1.]), k4=np.array([0.]),
        pair_source=np.array([0], np.int64), pair_target=np.array([1], np.int64), pair_keys=np.array([1], np.uint64),
        baseline_open=np.array([1.]), baseline_i95=np.array([2.]), c0=np.array([1.]))
    graph_file = npz(tmp_path/"graph.npz", **graph)
    identity = dict(phase_id="PACNHN_V1", run_id="synthetic", region_id=key,
        graph_sha256=graph_file["sha256"], graph_identity=a.array_hash({k:v for k,v in graph.items() if k != "c0"}),
        support_identity=a.array_hash({k:graph[k] for k in ("pair_source", "pair_target", "pair_keys")}))
    ib = put(root/"REGION_IDENTITY.json", identity); index = {}; records = []
    for at, value in enumerate((0., 2.)):
        u = np.array([value]); flow = u.copy(); input_id = a.array_hash({"u":u})
        folder = root/f"request_{at}"; nb = npz(folder/"label.npz", u=u, tau=np.array([1.]), c0=graph["c0"], pair_keys=graph["pair_keys"], flow=flow)
        record = dict(identity, input_identity=input_id, role="TRAIN", pair_count=1, quality="A", admitted=True,
            label_sha256=nb["sha256"], fresh_certificate=True, selected_state_hash=a.array_hash({"flow":flow}),
            certificate=dict(state_hash=a.array_hash({"flow":flow}), TSTT=value, SPTT=value,
                raw_signed_gap=0., RG_SPTT=0. if value else None, arithmetic_allowance=1e-10*max(1.,value),
                value=value, fresh=True, interpretation="CONVEX_POTENTIAL_FW_GAP"),
            potential_error_bound=1e-10*max(1.,value), potential_lower=value, potential_upper=value,
            marginal_cost_error_bound_available=False, last_cost_change=0., history=[{"irrelevant": "x"*10000}])
        rb = put(folder/"RESULT.json", record)
        index[input_id] = dict(role="TRAIN", selected_label=dict(path=nb["path"], sha256=nb["sha256"], receipt=rb["path"], receipt_sha256=rb["sha256"]))
        records.append(record)
    # Blind targets deliberately point at missing files. They must never open.
    index["f"*64] = dict(role="TEST", selected_label=dict(path=str(root/"FORBIDDEN_TEST.npz"), receipt=str(root/"FORBIDDEN_TEST.json")))
    xb = put(root/"REQUEST_INDEX.json", index)
    expected_data = data([[0.], [2.]], [[1.], [1.]])
    expected_data["c0"] = graph["c0"]
    expected = dict(key=key, rows=2, pairs=1,
        identity=a.array_hash({k:expected_data[k] for k in ("u", "tau", "c0")}))
    return root, dict(region_identity=ib, request_index=xb), graph, expected, index, records, graph_file


def test_train_loader_blind_test_and_discards_history(tmp_path):
    root, bindings, graph, expected, _, _, _ = label_fixture(tmp_path)
    reader = a.Reader(); result = a.load_train(reader, root, bindings, graph, expected, run_id="synthetic")
    assert result["dataset_identity"] == expected["identity"]
    assert len(result["records"]) == 2 and all("history" not in r for r in result["records"])
    assert not any("FORBIDDEN_TEST" in p for p in reader.bindings)
    summary = a.summarize_fresh(result)
    assert summary["quality_counts"] == {"A": 2, "B": 0}
    assert summary["structural_zero_proof"].startswith("NOT_REPLAYED")


def test_saved_c0_identity_preserved_with_producer_graph_tolerance(tmp_path):
    root, bindings, graph, expected, _, _, _ = label_fixture(tmp_path)
    graph["c0"] = graph["c0"] + 5e-12
    result = a.load_train(a.Reader(), root, bindings, graph, expected, run_id="synthetic")
    assert result["dataset_identity"] == expected["identity"]
    np.testing.assert_array_equal(result["c0"], [1.])
    assert not np.array_equal(result["c0"], graph["c0"])


def test_sealing_expiry_cannot_emit_verified_receipt(tmp_path):
    (tmp_path/"evidence.json").write_text('{"synthetic":true}')
    calls = []
    def expired():
        calls.append(1)
        if len(calls) == 5:
            raise TimeoutError("original shared deadline")
    with pytest.raises(TimeoutError, match="shared"):
        a._seal(tmp_path, expired)
    assert not (tmp_path/"SEAL.json").exists()


@pytest.mark.parametrize("mutation", ["role", "quality", "fresh", "identity", "certificate_state", "flow", "dataset", "index_hash"])
def test_train_loader_rejects_stale_or_invalid_selected_data(tmp_path, mutation):
    root, bindings, graph, expected, index, records, _ = label_fixture(tmp_path)
    key = next(iter(index)); selected = index[key]["selected_label"]
    if mutation == "dataset":
        expected["identity"] = "f"*64
    elif mutation == "index_hash":
        bindings["request_index"]["sha256"] = "0"*64
    elif mutation == "flow":
        with np.load(selected["path"], allow_pickle=False) as z:
            arrays = {k:z[k].copy() for k in z.files}
        arrays["flow"] += 1
        nb = npz(Path(selected["path"]), **arrays); selected["sha256"] = nb["sha256"]
        records[0]["label_sha256"] = nb["sha256"]
    else:
        field, val = {"role":("role","VALID"), "quality":("quality","B"), "fresh":("fresh_certificate",False),
                      "identity":("graph_identity","f"*64), "certificate_state":("selected_state_hash","f"*64)}[mutation]
        records[0][field] = val
    if mutation not in {"dataset", "index_hash"}:
        rb = put(Path(selected["receipt"]), records[0]); selected["receipt_sha256"] = rb["sha256"]
        bindings["request_index"] = put(root/"REQUEST_INDEX.json", index)
    with pytest.raises(ValueError):
        a.load_train(a.Reader(), root, bindings, graph, expected, run_id="synthetic")


def test_graph_identity_and_native_scale(tmp_path):
    root, bindings, graph, expected, index, records, gf = label_fixture(tmp_path)
    identity = json.loads(Path(bindings["region_identity"]["path"]).read_text())
    record = dict(gf, pairs=1)
    loaded = a.load_graph_identity(a.Reader(), record, identity)
    saved = dict(scale=[2.], fraction=.01, graph_sha256=gf["sha256"])
    np.testing.assert_array_equal(a.native_scale(loaded, record, saved, .01), [2.])
    with pytest.raises(ValueError, match="policy"):
        a.native_scale(loaded, record, saved, .1)
    identity["support_identity"] = "f"*64
    with pytest.raises(ValueError, match="identity"):
        a.load_graph_identity(a.Reader(), record, identity)


@pytest.mark.parametrize("mutation", ["expired", "extended", "wrong_stage", "early", "reset_end"])
def test_shared_clock_no_reset(mutation):
    clock = dict(schema="PACNHN_FINAL_AUDIT_CLOCK_V1", phase_id="PACNHN_V1", run_id="synthetic",
                 stage4_clock_sha256="a"*64, start_epoch=100., deadline_epoch=1900.)
    now = 110.
    if mutation == "expired": now = 1900.
    if mutation == "extended": clock["deadline_epoch"] += 1.
    if mutation == "wrong_stage": clock["stage4_clock_sha256"] = "b"*64
    if mutation == "early": clock["start_epoch"] = 50.
    if mutation == "reset_end": clock["deadline_epoch"] = 1800.
    with pytest.raises(ValueError):
        a.shared_window(clock, "a"*64, "synthetic", now, 100., 14500.)


def test_shared_clock_clamped_to_stage4():
    clock = dict(schema="PACNHN_FINAL_AUDIT_CLOCK_V1", phase_id="PACNHN_V1", run_id="synthetic",
                 stage4_clock_sha256="a"*64, start_epoch=100., deadline_epoch=1000.)
    assert a.shared_window(clock, "a"*64, "synthetic", 110., 100., 1000.) == 1000.


def test_exact1499_final_population_and_no_test_metrics(tmp_path):
    p = tmp_path/"LEARNING.csv"
    fields = ["suite_id", "region_index", "family", "current_train_dataset_identity", "train_rows", "pairs", "graph_sha256", "TEST_WMAPE_percent"]
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for i in range(1499):
            writer.writerow(dict(suite_id="s03", region_index=i, family="ICNN", current_train_dataset_identity="a"*64,
                                 train_rows=2, pairs=1, graph_sha256="b"*64, TEST_WMAPE_percent="not parsed"))
    rows = a.final_population(a.Reader(), binding(p))
    assert len(rows) == 1499 and all("TEST_WMAPE_percent" not in r for r in rows.values())
    with p.open("a") as f: f.write("s03,0,ICNN,"+"a"*64+",2,1,"+"b"*64+",ignored\n")
    with pytest.raises(ValueError, match="Duplicate"):
        a.final_population(a.Reader(), binding(p))


def test_sparse_active_region_ids_preserved_and_bound_to_support(tmp_path):
    ids=[i if i<1000 else i+348 for i in range(1499)]
    p=tmp_path/'sparse.csv'
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['suite_id','region_index','family','current_train_dataset_identity','train_rows','pairs','graph_sha256']);w.writeheader()
        for i in ids:w.writerow(dict(suite_id='s03',region_index=i,family='ICNN',current_train_dataset_identity='a'*64,train_rows=2,pairs=1,graph_sha256='b'*64))
    pop=a.final_population(a.Reader(),binding(p))
    support=dict(suites=[dict(suite_id='s03',active_regions=1499,support_frozen_before_labels=True,region_records=[dict(region_index=i) for i in ids])])
    regional=a.supported_regions(support)
    assert set(pop)==set(regional) and 's03:001846' in pop and 's03:001000' not in pop
    support['suites'][0]['region_records'][-1]=dict(region_index=0)
    with pytest.raises(ValueError,match='Duplicate/missing'):a.supported_regions(support)


def test_missing_stage4_rejects_before_label_access(tmp_path, monkeypatch):
    # Entry rejects incomplete explicit config even before Context or label load.
    p = tmp_path/"config.json"; put(p, dict(schema="wrong", phase_id="PACNHN_V1", step_id="final_training_limits",run_id="synthetic"))
    monkeypatch.setattr(a, "load_train", lambda *args, **kwargs: pytest.fail("No label access before admission"))
    with pytest.raises(ValueError, match="config"):
        a.run(p, admission_path=tmp_path/"missing", admission_sha256="0"*64)


def test_budget_checks_process_and_available_memory(monkeypatch):
    class Proc:
        def memory_info(self): return type("M", (), {"rss":100})()
        def children(self, **kwargs): return []
    class Ctx:
        def guard(self, **kwargs): pass
    monkeypatch.setattr(a.psutil, "Process", Proc)
    monkeypatch.setattr(a.psutil, "virtual_memory", lambda:type("V", (), {"available":100})())
    b = a.Budget(Ctx(), time.time()+10, dict(single_process_rss_bytes=50, tree_rss_bytes=1000, minimum_available_bytes=1))
    with pytest.raises(MemoryError, match="RSS"): b()
    b.resources["single_process_rss_bytes"] = 1000; b.resources["minimum_available_bytes"] = 101
    with pytest.raises(MemoryError, match="available"): b()
    b.deadline = time.time()-1
    with pytest.raises(TimeoutError): b()


def whole_audit_fixture(tmp_path, monkeypatch):
    """Full1499 metadata ledger, only four synthetic fresh label arrays."""
    from partC_rev1_src.pacnhn_v1 import handoff
    out = tmp_path/"out"; out.mkdir()
    now = time.time(); stage_start = now-20
    stage3clock = dict(original_stage_start_epoch=now-100)
    s3c = put(out/"STAGE3_CLOCK.json", stage3clock)
    s3 = put(out/"STAGE3_RESULT.json", dict(status="COMPLETE_WITH_RECORDED_OUTCOMES", primary_rows=14,
                                          clock=stage3clock, elapsed_seconds=60.))
    s4 = put(out/"STAGE4_CLOCK.json", dict(stage_start_epoch=stage_start, stage_deadline_epoch=stage_start+14400))
    amendment = put(out/"amendment.json", {"synthetic":True})
    shared = put(out/"final_audits/AUDIT_CLOCK.json", dict(schema="PACNHN_FINAL_AUDIT_CLOCK_V1", phase_id="PACNHN_V1",
        run_id="synthetic", stage4_clock_sha256=s4["sha256"], start_epoch=now-10, deadline_epoch=now-10+1800))
    fresh = {}; graph_records = {}; populations = {}
    for key in sorted(a.FRESH_KEYS):
        parent = out/"training/regions/s03"/key.split(":")[1]; parent.mkdir(parents=True)
        _, fb, _, pop, _, _, gb = label_fixture(parent, key=key)
        fb["native_scale"] = put(parent/"NATIVE_SCALE.json", dict(scale=[2.],fraction=.01,graph_sha256=gb["sha256"]))
        fresh[key] = fb; populations[key] = pop
        graph_records[key] = dict(gb,suite_id="s03",region_index=int(key.split(":")[1]),pairs=1)
    prep = out/"prep"; prep.mkdir()
    oldrows = []; learning_rows = []; region_rows = []
    for i in range(1499):
        key = f"s03:{i:06d}"; old_id = f"{i:064x}"
        pop = populations.get(key, dict(identity=old_id,rows=2,pairs=1))
        graph_record = graph_records.get(key, dict(suite_id="s03",region_index=i,pairs=1,sha256="e"*64,path="never read"))
        region_rows.append(graph_record)
        learning_rows.append(dict(suite_id="s03",region_index=i,family="ICNN",current_train_dataset_identity=pop["identity"],
                                 train_rows=2,pairs=1,graph_sha256=graph_record["sha256"]))
        receipt = dict(train_dataset_identity=old_id,screen=a.SCREEN,unique_rows=2,ordered_comparisons=0,
            violating_disjoint_pairs=0,violations=[],fixed_label_L1_error_lower_bound=0.,
            fixed_label_WMAPE_percent_lower_bound=0.,denominator_native_cost_sum=2.)
        rb = put(out/"training/regions/s03"/f"{i:06d}"/"monotonic_conflicts"/(old_id+".json"), receipt)
        oldrows.append(dict(region_key=key,receipt_path=rb["path"],receipt_sha256=rb["sha256"],saved_TRAIN_identity=old_id,
            final_TRAIN_identity=pop["identity"],saved_TRAIN_rows=2,final_TRAIN_rows=2))
    a._csv(out/"training/LEARNING_RESULTS.csv", learning_rows)
    terminal = put(out/"training/RESULT.json", dict(status="STAGE2_FINISHED_WITH_EXPLICIT_COVERAGE",coverage=[],
                    learning_csv_sha256=binding(out/"training/LEARNING_RESULTS.csv")["sha256"]))
    campaign = put(out/"campaign.json", dict(run_id="synthetic",output_root=str(out),
                                            continuation=dict(training_result_sha256=terminal["sha256"])))
    support = put(out/"SUPPORT_FREEZE.json", dict(status="PASS",suites=[dict(suite_id="s03",active_regions=1499,
                                              support_frozen_before_labels=True,region_records=region_rows)]))
    a._csv(prep/"CURRENT_RECEIPT_IDENTITIES.csv", oldrows)
    prepmanifest = put(prep/"ARTIFACT_MANIFEST.json", dict(artifacts=[dict(relative_path="CURRENT_RECEIPT_IDENTITIES.csv",
                         sha256=binding(prep/"CURRENT_RECEIPT_IDENTITIES.csv")["sha256"])]))
    inventory = put(out/"source_inventory.json", dict(files=[binding(Path(a.__file__).with_name(name)) for name in a.SOURCE_NAMES]
                    + [binding(Path(__file__))]))
    cfg = dict(schema="PACNHN_FINAL_TRAIN_LIMITS_CONFIG_V1",phase_id="PACNHN_V1",step_id="final_training_limits",run_id="synthetic",
        tasks=sorted(a.REQUIRED_TASKS),expected_outputs=["synthetic only"],acceptance_tests=["synthetic only"],input_versions={"synthetic":True},
        fresh_regions=sorted(a.FRESH_KEYS),resources=dict(wall_seconds=60,single_process_rss_bytes=1024**3,tree_rss_bytes=24*1024**3,
                                                       minimum_available_bytes=3*1024**3),
        campaign_config=campaign,stage3_result=s3,stage3_clock=s3c,stage4_clock=s4,clock_amendment=amendment,shared_audit_clock=shared,
        source_inventory=inventory,deadline_epoch=stage_start+14400,output_dir=str(out/"audits/real_synthetic_output"),
        training_result=terminal,support_freeze=support,preparation_manifest=prepmanifest,fresh_inputs=fresh)
    cb = put(out/"config.json", cfg)
    tests = put(out/"tests.json",dict(status="PASS",exit_code=0,scope="SYNTHETIC FIXTURE ONLY",
                 source_sha256=binding(Path(a.__file__))["sha256"],test_sha256=binding(Path(__file__))["sha256"]))
    admission = put(out/"admission.json",dict(schema="PACNHN_FINAL_TRAIN_LIMITS_ADMISSION_V1",decision="ADMIT_FINAL_TRAIN_LIMITS_AUDIT",
        phase_id="PACNHN_V1",run_id="synthetic",config_sha256=cb["sha256"],source_inventory_sha256=inventory["sha256"],
        stage4_clock_sha256=s4["sha256"],shared_audit_clock_sha256=shared["sha256"],stage3_result_sha256=s3["sha256"],tests_passed=True,test_receipt=tests))
    class Ctx:
        def __init__(self,*args,**kwargs): self.root=out
        def require_action(self, action): assert action == "analysis"
        def guard(self, **kwargs): pass
    class Budget:
        peak_rss=1000;peak_tree_rss=1000;minimum_available=4*1024**3
        def __init__(self,*args,**kwargs): pass
        def __call__(self): pass
    monkeypatch.setattr(a,"Context",Ctx);monkeypatch.setattr(a,"Budget",Budget)
    monkeypatch.setattr(handoff,"publication_window",lambda *args,**kwargs:dict(stage_start_epoch=stage_start,
                         stage_deadline_epoch=stage_start+14400,deadline_epoch=stage_start+14400))
    for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):monkeypatch.setenv(key,"1")
    return out, cb, admission


@pytest.mark.parametrize("fail_second", [False, True])
def test_full1499_ledger_and_retained_prefix(tmp_path, monkeypatch, fail_second):
    out, cfg, admission = whole_audit_fixture(tmp_path, monkeypatch)
    original = a.load_train
    def interrupted(reader, root, bindings, graph, expected, **kwargs):
        if fail_second and expected["key"] == "s03:000939":
            raise TimeoutError("synthetic shared audit cutoff")
        return original(reader, root, bindings, graph, expected, **kwargs)
    monkeypatch.setattr(a,"load_train",interrupted)
    result = a.run(cfg["path"],admission_path=admission["path"],admission_sha256=admission["sha256"])
    root = out/"audits/real_synthetic_output"
    with (root/"REGION_COVERAGE.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1499
    assert result["reused_regions"] == 1497
    if fail_second:
        assert result["status"] == "PARTIAL_BUDGET_OR_RESOURCE" and result["refreshed_regions"] == 1
        assert next(r for r in rows if r["region_key"] == "s03:000939")["disposition"] == "INCOMPLETE_BUDGET_OR_RESOURCE"
    else:
        assert result["status"] == "COMPLETE_WITH_SCOPE_LIMITATIONS" and result["refreshed_regions"] == 2
        assert result["completed_regions"] == 1499 and result["TRAIN_inputs"] == 2998
    assert (root/"SEAL.json").exists() and result["TEST_targets_opened"] is False
