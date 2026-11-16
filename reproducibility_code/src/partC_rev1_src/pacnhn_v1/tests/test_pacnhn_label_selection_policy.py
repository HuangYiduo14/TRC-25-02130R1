"""Pure config and extracted-AST adapter tests; no scientific imports or solves."""
import ast
import json
from pathlib import Path
import pytest
REPO=Path(__file__).resolve().parents[3]
R=REPO/'partC_rev1_src/pacnhn_v1'
def function(name,file):
    return next(n for n in ast.parse((R/file).read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name==name)
def helper():
    ns={};node=function('label_solver_settings','training.py')
    exec(compile(ast.Module(body=[node],type_ignores=[]),'isolated_pure_helper','exec'),ns)
    return ns['label_solver_settings']
def config():return json.loads((REPO/'partC_doc_icnn/pacnhn_v1/training_config_v4.json').read_text(encoding='utf-8'))
def test_v4_and_legacy_default_policy():
    c=config();h=helper();v4=h(c)
    assert v4['iterate_selection']=='LAST_CERTIFIED' and v4['retry_iterate_selection']=='BEST_CERTIFIED_RG'
    legacy=dict(c);legacy.pop('label_iterate_selection');legacy.pop('label_retry_iterate_selection')
    old=h(legacy);assert old['iterate_selection']==old['retry_iterate_selection']=='LAST_CERTIFIED'
    assert {k:v for k,v in old.items() if 'iterate_selection' not in k}=={k:v for k,v in v4.items() if 'iterate_selection' not in k}
    assert v4['max_attempts_per_input']==4
@pytest.mark.parametrize('value',[None,False,'BEST','last_certified'])
def test_invalid_selector_fails(value):
    c=config();c['label_retry_iterate_selection']=value
    with pytest.raises(ValueError):helper()(c)
@pytest.mark.parametrize('attempt,expected',[(1,'LAST_CERTIFIED'),(2,'BEST_CERTIFIED_RG'),(4,'BEST_CERTIFIED_RG')])
def test_cumulative_attempt_adapter_statements(attempt,expected):
    g=function('generate_labels','regional.py')
    loop=next(n for n in g.body if isinstance(n,ast.For) and isinstance(n.iter,ast.Name) and n.iter.id=='requests' and any(isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='resolved_selection' for t in x.targets) for x in n.body))
    wanted={'resolved_method','resolved_iterations','resolved_bound_reset','resolved_selection','attempt_policy'}
    nodes=[n for n in loop.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in wanted for t in n.targets)]
    ns=dict(attempt=attempt,solver_method='FW',retry_solver_method='CFW',first_iterations=2000,retry_iterations=20000,
        first_bound_reset=False,retry_bound_reset=True,first_selection='LAST_CERTIFIED',retry_selection='BEST_CERTIFIED_RG',policy={})
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'actual_dispatch_statements','exec'),ns)
    assert ns['attempt_policy']['resolved_iterate_selection']==expected
    assert ns['attempt_policy']['attempt_number_including_prior_failures']==attempt
    assert ns['resolved_method']==('FW' if attempt==1 else 'CFW')
    assert ns['resolved_iterations']==(2000 if attempt==1 else 20000)
def test_generator_forwards_to_frozen_solver_api_and_records_policies():
    g=function('generate_labels','regional.py');s=function('solve_label','regional.py')
    assert 'iterate_selection' in [a.arg for a in s.args.kwonlyargs]
    calls=[n for n in ast.walk(g) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='solve_label']
    assert len(calls)==1
    kw=next(k for k in calls[0].keywords if k.arg=='iterate_selection')
    assert isinstance(kw.value,ast.Name) and kw.value.id=='resolved_selection'
    policy=next(n for n in g.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='policy' for t in n.targets))
    assert {'first_iterate_selection','retry_iterate_selection'} <= {k.arg for k in policy.value.keywords}
