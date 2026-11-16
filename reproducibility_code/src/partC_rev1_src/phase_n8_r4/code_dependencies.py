"""Static local dependency closure for immutable code receipts, without imports."""
from pathlib import Path
import ast
from partC_rev1_src.phase_n8.common import sha


def dependencies(repo, roots):
    repo=Path(repo).resolve();queue=[Path(p) for p in roots];seen=set()
    while queue:
        p=queue.pop().resolve()
        if p in seen:continue
        if repo not in p.parents:raise ValueError('Dependency outside repository')
        seen.add(p);module=list(p.relative_to(repo).with_suffix('').parts)
        package=module[:-1]
        for parent in p.parents:
            if parent==repo:break
            init=parent/'__init__.py'
            if init.is_file() and init not in seen:queue.append(init)
        names=[]
        for node in ast.walk(ast.parse(p.read_text(encoding='utf-8-sig'))):
            if isinstance(node,ast.Import):names.extend(a.name for a in node.names)
            elif isinstance(node,ast.ImportFrom):
                base=package[:len(package)-node.level+1] if node.level else []
                if node.module:base+=node.module.split('.')
                names.append('.'.join(base))
                names.extend('.'.join(base+[a.name]) for a in node.names if a.name!='*')
        for name in names:
            if not name.startswith(('partC_rev1_src','tests','partC_rev1_tests')):continue
            path=repo/Path(*name.split('.'))
            for candidate in (path.with_suffix('.py'),path/'__init__.py'):
                if candidate.is_file() and candidate.resolve() not in seen:queue.append(candidate)
    return sorted(seen)


def physical_code_bindings(repo):
    repo=Path(repo)
    starts=[repo/'partC_rev1_src/phase_n8_r4/physical.py',
        repo/'partC_rev1_src/phase_n8_r4/physical_solver.py',Path(__file__)]
    starts+=list((repo/'partC_rev1_src/phase_n8_r4/tests').glob('*.py'))
    # The scoped tests reopen this fixture module by a filesystem path.
    starts.append(repo/'tests/test_phase4_preflight.py')
    return {p.relative_to(repo).as_posix():sha(p) for p in dependencies(repo,starts)}
