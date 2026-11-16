"""Narrow, backed-up roadmap updates; a user edit produces a proposed patch."""
from pathlib import Path
import difflib
import shutil
import subprocess
from partC_rev1_src.phase_n8.common import read_json,write_json,sha
from .reporting import REPORT_NAMES


def sync(ctx,args):
    root=ctx.root;repo=ctx.repo;state=read_json(root/'02_HANDOFF_STATUS.json')
    docs=Path(ctx.c['documentation_root']);docs.mkdir(parents=True,exist_ok=True)
    if repo not in docs.parents:raise ValueError('Versioned documentation must be scoped inside the configured repository')
    marker='<!-- N8R4_NUM_CURRENT_BEGIN -->';end='<!-- N8R4_NUM_CURRENT_END -->'
    relative=docs.relative_to(repo/'partC_doc_icnn').as_posix()
    block=(marker+'\n# Current N8R4 numerical round\n\n'
        f'The user explicitly launched `{ctx.c["authority_id"]}`, ZIP SHA-256 `{ctx.c["authority_sha256"]}`, '
        f'on the existing `phase1/icnn-rev` branch. This authority supersedes historical R4-not-authorized wording only within the bounded N8R4 numerical scope. '
        f'Current execution: **{state["execution"]}**; review status: **{state["review_status"]}**.\n\n'
        f'Read [the versioned R4 report]({relative}/00_READ_FIRST.md), the exact authority pointer and maintained run evidence. '
        'The original R4 clock is never reset; the final four hours are reserved for reporting. '
        's01 is excluded; original source models, accepted active support and baseline results remain protected. '
        'The user authorized one integrated final handoff, with no per-substep stop inside this round. '
        '**No next round is authorized.** Historical decisions below remain unchanged as provenance.\n\n'+end+'\n\n')
    receipt_path=root/'receipts/DOCUMENTATION_SYNC.json'
    previous=read_json(receipt_path) if receipt_path.exists() else {'files':[]}
    owned={r['relative_path']:r for r in previous['files'] if r.get('status')=='UPDATED'}
    backup=root/'documentation_backups';backup.mkdir(exist_ok=True);records=[]
    for name in ('00_READ_FIRST.md','PROJECT_STATUS.md','12_DECISION_LOG.md'):
        path=repo/'partC_doc_icnn'/name;rel=path.relative_to(repo).as_posix();current=path.read_text(encoding='utf-8')
        if rel in owned:clean=sha(path)==owned[rel]['current_sha256']
        else:
            command=['git','-C',str(repo),'-c','core.longpaths=true','status','--porcelain','--',rel]
            result=subprocess.run(command,text=True,capture_output=True,check=True)
            clean=not result.stdout.strip()
        if marker in current:
            if not current.startswith(marker) or current.count(marker)!=1 or current.count(end)!=1:
                raise ValueError('Ambiguous current R4 documentation block')
            old_tail=current.split(end,1)[1].lstrip('\n')
        else:old_tail=current
        revised=block+old_tail
        original=backup/name
        if clean:
            if not original.exists():shutil.copy2(path,original)
            path.write_text(revised,encoding='utf-8')
            records.append(dict(relative_path=rel,status='UPDATED',original_backup=str(original),
                original_sha256=sha(original),current_sha256=sha(path)))
        else:
            patch=backup/(name+'.PROPOSED.patch')
            patch.write_text(''.join(difflib.unified_diff(current.splitlines(True),revised.splitlines(True),
                fromfile='a/'+rel,tofile='b/'+rel)),encoding='utf-8')
            records.append(dict(relative_path=rel,status='USER_DIRTY_PROPOSED_PATCH_ONLY',proposed_patch=str(patch),
                preserved_current_sha256=sha(path)))
    report_copies=[]
    for name in REPORT_NAMES:
        if (root/name).exists():
            shutil.copy2(root/name,docs/name)
            report_copies.append(dict(name=name,source_sha256=sha(root/name),
                relative_path=(docs/name).relative_to(repo).as_posix(),copied_sha256=sha(docs/name)))
    write_json(docs/'AUTHORITY_POINTER.json',dict(authority_id=ctx.c['authority_id'],
        authority_root=str(ctx.authority),authority_sha256=ctx.c['authority_sha256'],output_root=str(root),
        run_id=ctx.c['run_id'],original_start_epoch=ctx.c['start_epoch'],compute_cutoff_epoch=ctx.c['compute_cutoff_epoch'],
        deadline_epoch=ctx.c['deadline_epoch'],authority_documents=[str(p) for p in sorted(ctx.authority.glob('[0-9][0-9]_*.md'))],
        launch_prompt_receipt=str(root/'receipts/USER_AUTHORITY_RECORD.json'),next_round_authorized=False))
    result=dict(status='EXECUTED',files=records,versioned_documentation_root=str(docs),
        report_copies=report_copies,source_handoff_status_sha256=sha(root/'02_HANDOFF_STATUS.json'),
        review_status=state['review_status'],execution=state['execution'],
        source_results_remain_external=True,next_round_authorized=False)
    write_json(receipt_path,result);return result
