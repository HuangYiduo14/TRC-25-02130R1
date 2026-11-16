"""Component-specific code/config/input bindings for automatic technical gates."""
from pathlib import Path
from .control import read_json, sha

COMMON = ['control.py','continuation.py','oracle.py','solver.py','topology.py','cli.py','gates.py']
COMPONENTS = {
    'reference': [],
    'support': ['support.py','witnesses.py'],
    'pilot': ['pilots.py','assignments.py','registry_projection.py','objective.py','models.py'],
    'inference': ['inference_pilot.py','assignments.py','registry_projection.py','objective.py','models.py'],
    'training': ['training.py','models.py','regional.py'],
    'assignments': ['assignments.py','registry_projection.py','objective.py','models.py'],
    'analysis': ['analysis_export.py','metrics.py','timing.py','final_icnn_guarantees.py',
                 'final_training_limits.py','final_assignment_metrics.py','models.py',
                 'registry_projection.py','regional.py','training.py','handoff.py'],
    'publication': ['publication.py','publication_evidence.py','publication_results.py','publication_training_curves.py','handoff.py','publication_templates/plot.py',
                    'publication_templates/build_table.py','publication_templates/audit_runner.py',
                    'publication_templates/csv_support.py'],
    'handoff': ['handoff.py'],
}


def component_snapshot(ctx, component):
    if component not in COMPONENTS:raise ValueError('Unknown technical-gate component')
    base=Path(__file__).parent
    files=[base/name for name in COMMON+COMPONENTS[component]]
    # Preserve and authenticate the inherited numerical helpers too. Including
    # the small whole helper packages is conservative against indirect imports.
    for package in ('phase4','phase5','phase_n8'):
        files.extend((ctx.repo/'partC_rev1_src'/package).glob('*.py'))
    files.extend([ctx.repo/'partC_rev1_src/partC_rev1_assignment.py',
                  ctx.repo/'partC_rev1_src/partC_rev1_partition_igraph.py'])
    return {str(p.resolve()):sha(p) for p in sorted(set(files))}


def topology_snapshot(ctx):
    root=ctx.root/'topology';path=root/'RESULT.json'
    if not path.exists():return None
    r=read_json(path)
    if r['status']!='PASS':raise ValueError('Topology hard gate is not PASS')
    result={k:r[k] for k in ('graph_identity','OD_identity','graph_array_identity','OD_array_identity','partitions_identity')}
    result['file_hashes']={str(root/name):sha(root/name) for name in
        ('RESULT.json','common_graph.npz','common_od.npz','partitions.npz')}
    return result


def require_component(ctx, component, *, extra_config=None):
    if getattr(ctx, 'c', {}).get('continuation'):
        ctx.require_action(component)
    record=read_json(ctx.root/'validation/PREFLIGHT_LATEST.json')
    if record.get('status')!='PASS' or record.get('config_sha256')!=sha(ctx.path):
        raise ValueError('Passing current technical preflight required')
    if record.get('component_snapshots',{}).get(component)!=component_snapshot(ctx,component):
        raise ValueError(component+' code changed since preflight; rerun preflight')
    if record.get('topology_snapshot')!=topology_snapshot(ctx):
        raise ValueError('Topology arrays/OD/partition inputs changed since preflight')
    if extra_config is not None:
        path=Path(extra_config).resolve()
        if record.get('extra_configs',{}).get(str(path))!=sha(path):
            raise ValueError('Explicit component configuration changed since preflight')
    return record
