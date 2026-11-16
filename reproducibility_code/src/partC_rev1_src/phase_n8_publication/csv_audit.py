"""Independent authority CSV arithmetic; no project numerical kernel reuse."""
from pathlib import Path
import math
import pandas as pd
from partC_rev1_src.phase_n8.common import write_json,sha
from . import authority_csv_check as independent


def run(ctx,qa_root):
    dst=Path(qa_root)/'csv_arithmetic';dst.mkdir(exist_ok=False);records=[];summaries=[]
    expected=pd.read_csv(ctx.root/'metrics/open_summary_metrics.csv',float_precision='round_trip')
    closed_expected=pd.read_csv(ctx.root/'metrics/closed_summary_metrics.csv',float_precision='round_trip')
    for sid,state in [(sid,'OPEN') for sid in ctx.c['active_suites']]+[('s03','CLOSED')]:
        reference=ctx.root/'reference'/sid/'backbone_reference.csv' if state=='OPEN' else ctx.root/'metrics/s03/closed_backbone_reference.csv'
        key=ctx.root/'reference'/sid/'key_links_200.csv';refmethod='REFERENCE_X45' if state=='OPEN' else 'REFERENCE_CLOSED'
        ref=independent.read_csv(reference,independent.REF_FIELDS);keys=independent.read_csv(key,independent.KEY_FIELDS)
        if len(keys)!=200:raise ValueError('CSV checker requires exactly frozen200')
        preds=[]
        for p in sorted((ctx.root/'metrics'/sid).glob('*_'+state.lower()+'_per_link.csv')):
            if p.name.startswith(refmethod+'_'):continue
            preds.extend(independent.read_csv(p,independent.PRED_FIELDS))
        result,per=independent.compute(ref,preds,keys,reference_method_id=refmethod)
        target=expected if state=='OPEN' else closed_expected
        coefficients=pd.read_csv(ctx.root/'metrics'/sid/'backbone_cost_coefficients.csv',float_precision='round_trip').set_index('abstract_link_id')
        for row in result:
            match=target.loc[target.suite_id.eq(sid)&target.method_id.eq(row['method_id'])&target.population.eq(row['population'])]
            if len(match)!=1:raise ValueError('CSV summary row identity differs')
            t=match.iloc[0]
            for col in ('link_count','reference_flow_sum','absolute_error_sum','wmape_percent','mae','rmse','mean_signed_error','reference_zero_link_count'):
                if row[col]=='':
                    if pd.notna(t[col]):raise ValueError('Undefined CSV denominator is not reported as missing')
                elif not math.isclose(float(row[col]),float(t[col]),rel_tol=5e-12,abs_tol=1e-8):raise ValueError('Independent CSV arithmetic differs: '+sid+'/'+row['method_id']+'/'+col)
            records.append(dict(suite_id=sid,state=state,method_id=row['method_id'],population=row['population'],status='PASS',reference_sha256=sha(reference),fixed_keys_sha256=sha(key)))
            summaries.append(dict(row,network_state=state))
        # Independently recompute fixed-charge backbone cost and closed-key zeros.
        frame=pd.DataFrame(per)
        for method,g in frame.groupby('method_id'):
            g=g.set_index('abstract_link_id');cf=coefficients.loc[g.index]
            rc=math.fsum(float(x)*(float(b)+float(k)*float(x)**4) for x,b,k in zip(g.lambda_reference,cf.base_seconds,cf.k4))
            pc=math.fsum(float(x)*(float(b)+float(k)*float(x)**4) for x,b,k in zip(g.lambda_estimate,cf.base_seconds,cf.k4))
            t=target.loc[target.suite_id.eq(sid)&target.method_id.eq(method)&target.population.eq('BACKBONE')].iloc[0]
            if not math.isclose(100*abs(pc-rc)/rc,float(t.backbone_cost_error_percent),rel_tol=1e-11,abs_tol=1e-8):raise ValueError('Independent packed backbone cost differs')
        print('INDEPENDENT_CSV_METRICS_PASS',sid,state,flush=True)
    pd.DataFrame(records).to_csv(dst/'CHECKS.csv',index=False);pd.DataFrame(summaries).to_csv(dst/'INDEPENDENT_SUMMARIES.csv',index=False,float_format='%.17g')
    result=dict(status='PASS',aggregate_rows=len(records),evidence_directory=str(dst),checker_sha256=sha(Path(independent.__file__)),
        adaptation='Only the open reference requires key reference flow >0. Closed reference retains exactly the pre-frozen baseline200 including newly zero flows.',
        construction_scope='Arithmetic validation only; complete native backbone/mapping proof is separately required and recorded.')
    write_json(dst/'RESULT.json',result);return result
