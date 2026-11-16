"""Build and verify the CSV-only release, then compile and render its manuscript."""
from pathlib import Path
import ast
import os
import json
import shutil
import subprocess
import sys
import time
import re
import zipfile
import numpy as np
import pandas as pd
from partC_rev1_src.phase_n8.common import read_json,write_json,sha

ASSETS=['figure'+str(i) for i in range(3,9)]+['figureS'+str(i) for i in range(1,5)]+['table'+str(i) for i in range(1,7)]+['tableS1']


def build(ctx,args):
    from .publication_data import Release,primary,regions,supplements
    from .publication_tables import build as tables
    from .publication_manuscript import write
    if read_json(ctx.root/'metrics/RESULT.json')['status']!='PASS':raise ValueError('Native-coordinate metrics gate incomplete')
    release=Release(ctx);summary,suites=primary(release);ctx.guard()
    costs,fits,coverage=regions(release);ctx.guard();fallback=supplements(release);ctx.guard()
    facts=tables(release,summary,suites,costs,fits,coverage,fallback);ctx.guard();text=write(release,facts)
    release.finish()
    if set(release.files)!=set(ASSETS):raise ValueError('Publication folder matrix incomplete')
    write_json(release.root/'EXTRACTION_RESULT.json',dict(status='PASS',assets=ASSETS,text=text,
        document_complete=False,rendering_pending=True,native_predicted_physical_recovery=False))
    return dict(status='EXTRACTED',assets=len(ASSETS),document_complete=False)


def scan_script(path):
    tree=ast.parse(path.read_text(encoding='utf-8'));imports=[]
    allowed={'pathlib','csv','hashlib','textwrap','numpy','pandas','matplotlib'}
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):imports.extend(a.name for a in node.names)
        if isinstance(node,ast.ImportFrom):imports.append(node.module or '')
        if isinstance(node,ast.Name) and node.id in {'eval','exec','__import__','compile','subprocess','importlib'}:
            raise ValueError('Dynamic/external drawing code: '+node.id)
        if isinstance(node,ast.Constant) and isinstance(node.value,str):
            if re.search(r'https?://|[A-Z]:[\\/]|\.npz|\.parquet|\.sqlite|sys\.path',node.value,re.I):
                raise ValueError('External drawing data locator: '+node.value)
    if any(m.split('.')[0] not in allowed for m in imports):raise ValueError('Disallowed drawing imports')
    return imports


AUDIT_DRIVER=r'''
import sys,os,json,runpy
from pathlib import Path
root=Path(sys.argv[1]).resolve();repo=Path(sys.argv[2]).resolve();script=sys.argv[3];events=[]
prefix=Path(sys.prefix).resolve();base=Path(sys.base_prefix).resolve();system=Path(os.environ.get('WINDIR','C:/Windows')).resolve()
def inside(p,r):return p==r or r in p.parents
def audit(event,args):
    if event!='open' or not args or isinstance(args[0],int):return
    try:p=Path(args[0]).resolve()
    except (TypeError,ValueError):return
    mode=args[1] if len(args)>1 else None
    writing=isinstance(mode,str) and any(c in mode for c in 'wax+')
    if inside(p,repo):raise RuntimeError('Repository access prohibited: '+str(p))
    if writing:
        if not inside(p,root):raise RuntimeError('Write outside isolated folder: '+str(p))
        return
    if inside(p,root):
        if p==root/script or inside(p,root/'.mplcache'):return
        if p.parent==root and p.suffix=='.csv':events.append(str(p.relative_to(root)));return
        raise RuntimeError('Non-CSV drawing input: '+str(p))
    if inside(p,prefix) or inside(p,base) or inside(p,system):return
    raise RuntimeError('External data read: '+str(p))
sys.addaudithook(audit)
runpy.run_path(str(root/script),run_name='__main__')
origins=[]
for name,module in sorted(sys.modules.items()):
    origin=getattr(module,'__file__',None)
    if not origin or str(origin).startswith('<'):continue
    p=Path(origin).resolve()
    if not any(inside(p,r) for r in (root,prefix,base,system)):
        raise RuntimeError('Imported module outside isolated/runtime roots: '+name+' '+str(p))
    origins.append({'module':name,'origin':str(p)})
versions={name:getattr(sys.modules.get(name),'__version__',None) for name in ('numpy','pandas','matplotlib')}
(root/'runtime_file_access.json').write_text(json.dumps({'csv_reads':sorted(set(events)),
    'repository_access':False,'external_data_access':False,'module_origins':origins,
    'package_versions':versions,'python_version':sys.version,'module_origins_validated':True},indent=2))
'''


def execute(ctx,command,cwd,env,log,cap=900):
    ctx.guard()
    with Path(log).open('w',encoding='utf-8') as f:
        result=subprocess.run(command,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,
            timeout=min(cap,max(1,ctx.deadline-time.time()-60)))
    if result.returncode:raise RuntimeError('Publication command failed: '+str(log))


def render_and_isolate(ctx,args):
    data=ctx.root/'publication_bundle/numerical_data';qa=ctx.root/'validation/publication'/args.job_id;qa.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,PYTHONPATH='',PYTHONDONTWRITEBYTECODE='1',MPLBACKEND='Agg',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    results=[]
    contracts=pd.read_csv(ctx.authority/'config/figures.csv')
    for row in contracts.itertuples():
        folder=ctx.root/'publication_bundle'/row.directory
        for filename in row.input_csv_contract.split(';'):
            stem=Path(filename).stem
            if not (folder/filename).is_file() and not list(folder.glob(stem+'_*.csv')):
                raise ValueError('Required local CSV input contract missing: '+str(folder/filename))
    for asset in ASSETS:
        folder=data/asset;script='build_table.py' if asset.startswith('table') else 'plot.py';imports=scan_script(folder/script)
        command=[sys.executable,'-I','-B',str(folder/script)];localenv=dict(env,MPLCONFIGDIR=str(folder/'.mplcache'))
        execute(ctx,command,qa,localenv,qa/(asset+'_production.log'))
        pdf='table.pdf' if asset.startswith('table') else asset+'.pdf';png='table.png' if asset.startswith('table') else asset+'.png'
        for name in (pdf,png,'read_csv_hashes.csv'):
            if not (folder/name).is_file() or not (folder/name).stat().st_size:raise ValueError('Missing rendered '+asset+'/'+name)
        isolated_records=[]
        for run in ('A','B'):
            alone=qa/(asset+'_alone_'+run);cwd=qa/('cwd_'+asset+'_'+run);cwd.mkdir()
            shutil.copytree(folder,alone,ignore=shutil.ignore_patterns('*.pdf','*.png','table.tex','.mplcache','read_csv_hashes.csv','runtime_file_access.json'))
            cmd=[sys.executable,'-I','-B','-c',AUDIT_DRIVER,str(alone),str(ctx.repo),script]
            execute(ctx,cmd,cwd,dict(env,MPLCONFIGDIR=str(alone/'.mplcache')),qa/(asset+'_isolated_'+run+'.log'))
            if sha(folder/'read_csv_hashes.csv')!=sha(alone/'read_csv_hashes.csv'):raise ValueError('Input rows/hashes differ in isolation: '+asset)
            # PNG byte equality excludes a dependency on neighboring files or cwd.
            for p in folder.glob('*.png'):
                if sha(p)!=sha(alone/p.name):raise ValueError('Rendered pixels differ in isolation: '+asset+'/'+p.name)
            if asset.startswith('table') and sha(folder/'table.tex')!=sha(alone/'table.tex'):raise ValueError('Table TeX differs in isolation')
            isolated_records.append(dict(run=run,cwd=str(cwd),runtime_access=read_json(alone/'runtime_file_access.json'),
                empty_PYTHONPATH=True,generated_inputs_removed=True,read_hashes_equal=True,PNG_equal=True))
        verified_files={p.name:sha(p) for p in sorted(folder.iterdir()) if p.is_file()}
        results.append(dict(asset=asset,status='PASS',imports=imports,source_script_sha256=sha(folder/script),
            verified_files=verified_files,runs=isolated_records))
        write_json(qa/'ISOLATION_RESULT.json',dict(status='IN_PROGRESS',assets=results))
        print('PUBLICATION_CSV_ISOLATION_PASS',asset,flush=True)
    write_json(qa/'ISOLATION_RESULT.json',dict(status='PASS',assets=results,standalone_folders=17,distinct_cwd_per_asset=2))
    return qa,results


def toolchain(ctx):
    root=ctx.root/'toolchain';root.mkdir(exist_ok=True);exe=root/'tectonic.exe'
    old=ctx.r4/'toolchain';ctx.bind(old/'tectonic.exe','USER_APPROVED_PORTABLE_LATEX_COMPILER')
    if not exe.exists():shutil.copy2(old/'tectonic.exe',exe)
    if sha(exe)!=sha(old/'tectonic.exe'):raise ValueError('Compiler copy differs')
    if not (root/'cache').exists():shutil.copytree(old/'cache',root/'cache')
    return exe,dict(os.environ,TECTONIC_CACHE_DIR=str(root/'cache'),XDG_CACHE_HOME=str(root/'cache'),XDG_CONFIG_HOME=str(root/'config'))


def compile_and_render(ctx,qa):
    root=ctx.root/'publication_bundle';exe,env=toolchain(ctx)
    command=[str(exe),'--only-cached','--keep-logs','--keep-intermediates','--outdir',str(root),str(root/'main_numerical.tex')]
    # The user approved this isolated portable toolchain. Bootstrap any missing
    # TeX bundle members only inside its cache, then prove cached recompilation.
    for round in (1,2):execute(ctx,[x for x in command if round!=1 or x!='--only-cached'],root,env,qa/('tectonic_pass_'+str(round)+'.log'),cap=1800)
    pdf=root/'main_numerical.pdf'
    base=Path(os.environ['USERPROFILE'])/'.cache/codex-runtimes/codex-primary-runtime/dependencies'
    poppler=base/'native/poppler/Library/bin/pdftoppm.exe';python=base/'python/python.exe'
    textpath=qa/'main_numerical.txt'
    script="from pathlib import Path;from pypdf import PdfReader;import sys,json;r=PdfReader(sys.argv[1]);Path(sys.argv[2]).write_text('\\n\\f\\n'.join(p.extract_text(extraction_mode='layout') or '' for p in r.pages),encoding='utf-8');print(json.dumps({'pages':len(r.pages)}))"
    execute(ctx,[str(python),'-I','-B','-c',script,str(pdf),str(textpath)],qa,env,qa/'PDF_TEXT_EXTRACTION.json',120)
    text=textpath.read_text(encoding='utf-8');log=(root/'main_numerical.log').read_text(encoding='utf-8',errors='replace')
    if any(x in text for x in ('??','[?]','@@')):raise ValueError('Unresolved references/placeholders in compiled PDF')
    issues=[line for line in log.splitlines() if re.search(r'Overfull|undefined|not found|Missing character',line,re.I)]
    (qa/'layout_warnings.txt').write_text('\n'.join(issues),encoding='utf-8')
    benign_info=r'Package pdftexcmds Info: \pdfdraftmode not found.'
    if any(re.search(r'undefined|not found|Missing character',line,re.I) for line in issues if line.strip()!=benign_info):raise ValueError('Undefined reference or missing output/font in PDF')
    pages=qa/'pdf_pages';pages.mkdir();execute(ctx,[str(poppler),'-scale-to','1600','-png',str(pdf),str(pages/'page')],qa,env,qa/'PDF_RENDER.log',900)
    # Every standalone table PDF page is rendered too; contact sheets aid actual visual inspection.
    from PIL import Image,ImageOps,ImageDraw
    panels=[]
    for asset in ASSETS:
        folder=root/'numerical_data'/asset
        panels.extend(folder.glob('*.png'))
    paneldir=qa/'asset_contact_sheets';paneldir.mkdir()
    for name,images,output,tile in [('pages',sorted(pages.glob('page-*.png')),qa,(900,1250)),('assets',panels,paneldir,(1200,850))]:
        for begin in range(0,len(images),4):
            canvas=Image.new('RGB',(2*tile[0],2*tile[1]),'#dce1e5');draw=ImageDraw.Draw(canvas)
            for j,p in enumerate(images[begin:begin+4]):
                with Image.open(p) as im:t=ImageOps.contain(im.convert('RGB'),(tile[0]-20,tile[1]-45))
                x=(j%2)*tile[0]+(tile[0]-t.width)//2;y=(j//2)*tile[1]+30;canvas.paste(t,(x,y));draw.text(((j%2)*tile[0]+10,(j//2)*tile[1]+8),p.parent.name+'/'+p.name,fill='black')
            canvas.save(output/(name+'_contact_'+str(begin//4+1)+'.png'))
    tex_sources={name:sha(root/name) for name in ('main_numerical.tex','numerical_section.tex','numerical_supplement.tex','results_macros.tex')}
    write_json(qa/'PDF_BUILD.json',dict(status='COMPILED_TWICE_AND_RENDERED',pages=len(list(pages.glob('page-*.png'))),pdf_sha256=sha(pdf),
        tex_source_hashes=tex_sources,layout_warnings=issues,visual_inspection_pending=True))
    return dict(status='COMPILED',main_pdf_sha256=sha(pdf),visual_inspection_pending=True)


def v5_review_copy(ctx,qa):
    """Replace only the numerical section in a visibly marked derivative copy."""
    root=ctx.root/'publication_bundle';archive=ctx.authority/'references/v5_reference.zip'
    ctx.bind(archive,'EXACT_V5_ORGANIZATIONAL_REFERENCE','c3fba115c03989762b7a08fe8dd86a7aed002aae2b279de2d2080b26a29e36b4')
    dest=root/'v5_review_source'
    if dest.exists():
        shutil.copytree(dest,qa/'prior_v5_review_source')
        for path in root.glob('NUMERICAL_REPLACEMENT_REVIEW_COPY.*'):
            if path.is_file():shutil.copy2(path,qa/('prior_'+path.name))
    dest.mkdir(exist_ok=True);manifest=[]
    assets=['Figure_2_v4.pdf','v5_new_drawing_table_code/data/fig7_1_mainland.png','v5_new_drawing_table_code/data/fig7_1_chicago.png']
    with zipfile.ZipFile(archive) as z:
        names=[n for n in z.namelist() if '/' not in n and n.endswith(('.tex','.cls','.bst','.bib')) and n not in ('main.tex','numerical example.tex')]+assets
        written_names=set()
        for name in names:
            # Flatten this small, explicitly selected reference set; never extract arbitrary paths.
            data=z.read(name);out=dest/Path(name).name
            if out.name in written_names:raise ValueError('Duplicate flattened v5 member')
            written_names.add(out.name)
            out.write_bytes(data);manifest.append(dict(zip_member=name,relative_path=out.relative_to(root).as_posix(),original_member_sha256=sha(out)))
        main=z.read('main.tex').decode('utf-8-sig')
    model=dest/'model.tex';text=model.read_text(encoding='utf-8-sig')
    text=text.replace('v5_new_drawing_table_code/generated_panels/Figure_1a_national_partition.pdf','v5_review_source/fig7_1_mainland.png')
    text=text.replace('v5_new_drawing_table_code/generated_panels/Figure_1b_chicago_hyperedge.pdf','v5_review_source/fig7_1_chicago.png')
    text=text.replace('{Figure_2_v4.pdf}','{v5_review_source/Figure_2_v4.pdf}')
    model.write_text(text,encoding='utf-8')
    for row in manifest:row['derivative_sha256']=sha(root/row['relative_path'])
    for name in ('title page','highlights','introduction','literatures','model','GB','conclusion','statements'):
        main=main.replace('\\input{'+name+'}','\\input{v5_review_source/'+name+'}')
    main=main.replace(r'\input{numerical example}',r'\clearpage\nolinenumbers\input{numerical_section}')
    main=main.replace(r'\bibliography{cas-refs}',r'\bibliography{v5_review_source/cas-refs}')
    main=main.replace(r'\bibliographystyle{elsarticle-harv}',r'\bibliographystyle{v5_review_source/elsarticle-harv}')
    banner=r'\AddToShipoutPictureFG{\AtPageUpperLeft{\put(24,-18){\makebox(0,0)[lt]{\color{red!65!black}\sffamily\fontsize{7}{8}\selectfont REVIEW COPY: only numerical Section 5 and supplement are current; all other text is unrevised.}}}}'
    main=main.replace(r'\begin{document}',r'\usepackage{pdflscape,textcomp,xcolor,eso-pic}\input{results_macros}'+'\n'+r'\begin{document}'+'\n'+banner)
    notice=r'''\clearpage\noindent\fbox{\parbox{.94\linewidth}{\textbf{Numerical replacement review copy.} Only the numerical section and its numerical supplement are replaced. The abstract, highlights, introduction, model/theory, conclusion and statements are historical and have not been reconciled with the new results. Historical Figure 1 is rendered from the exact supplied background rasters. Any old numerical claims or repository URL placeholders outside Section 5 remain unrevised.}}\par\clearpage
'''
    main=main.replace(r'\end{frontmatter}',r'\end{frontmatter}'+'\n'+notice)
    main=main.replace(r'\end{document}',r'\clearpage\input{numerical_supplement}\end{document}')
    (root/'NUMERICAL_REPLACEMENT_REVIEW_COPY.tex').write_text(main,encoding='utf-8')
    (root/'V5_REVIEW_COPY_SCOPE.md').write_text('# Numerical replacement review copy\n\nOnly Section 5 and its numerical supplement are current. A notice is repeated on every page. Abstract, highlights, introduction, model/theory, conclusion, statements and repository URL placeholders remain stale and must be reconciled before submission. The exact original ZIP is unchanged. Historical Figure 1 uses the supplied background rasters because its named generated PDFs are absent; these rasters are not inputs to any numerical_data drawing. Historical Figure 2 retains its supplied diagram, including inherited labels crossing box boundaries; it was not redrawn under this numerical-only amendment. The historical frontmatter also retains its orphaned keyword continuation. These are review-copy limitations, not defects in the current standalone numerical manuscript.\n',encoding='utf-8')
    # elsarticle class must resolve in the compiler working directory; this exact
    # supplied class is a manuscript dependency, not a drawing dependency.
    shutil.copy2(dest/'elsarticle.cls',root/'elsarticle.cls')
    exe,env=toolchain(ctx);cmd=[str(exe),'--only-cached','--keep-logs','--keep-intermediates','--outdir',str(root),str(root/'NUMERICAL_REPLACEMENT_REVIEW_COPY.tex')]
    try:
        for i in (1,2):execute(ctx,[x for x in cmd if i!=1 or x!='--only-cached'],root,env,qa/('v5_review_compile_'+str(i)+'.log'),cap=900)
        pdf=root/'NUMERICAL_REPLACEMENT_REVIEW_COPY.pdf'
        base=Path(os.environ['USERPROFILE'])/'.cache/codex-runtimes/codex-primary-runtime/dependencies'
        poppler=base/'native/poppler/Library/bin/pdftoppm.exe'
        pages=qa/'v5_pdf_pages';pages.mkdir()
        execute(ctx,[str(poppler),'-scale-to','1600','-png',str(pdf),str(pages/'page')],qa,env,qa/'V5_PDF_RENDER.log',900)
        from PIL import Image,ImageOps,ImageDraw
        images=sorted(pages.glob('page-*.png'));tile=(900,1250)
        for begin in range(0,len(images),4):
            canvas=Image.new('RGB',(1800,2500),'#dce1e5');draw=ImageDraw.Draw(canvas)
            for j,path in enumerate(images[begin:begin+4]):
                with Image.open(path) as im:thumb=ImageOps.contain(im.convert('RGB'),(880,1205))
                x=(j%2)*tile[0]+(tile[0]-thumb.width)//2;y=(j//2)*tile[1]+30
                canvas.paste(thumb,(x,y));draw.text(((j%2)*tile[0]+10,(j//2)*tile[1]+8),'v5 review/'+path.name,fill='black')
            canvas.save(qa/('v5_pages_contact_'+str(begin//4+1)+'.png'))
        result=dict(status='COMPILED_REVIEW_COPY',pdf_sha256=sha(pdf),pages=len(images),
            tex_sha256=sha(root/'NUMERICAL_REPLACEMENT_REVIEW_COPY.tex'),visual_inspection_pending=True)
    except Exception as exc:
        unverified=root/'NUMERICAL_REPLACEMENT_REVIEW_COPY.pdf'
        if unverified.exists():
            # Preserve a failed optional derivative outside the public release.
            # Its presence must never bypass the current PDF visual-QA gate.
            shutil.move(str(unverified),str(qa/'UNVERIFIED_NUMERICAL_REPLACEMENT_REVIEW_COPY.pdf'))
        result=dict(status='SOURCE_REVIEW_COPY_ONLY',error=str(exc),mandatory_numerical_only_PDF_unaffected=True)
    write_json(root/'private_provenance/V5_REVIEW_COPY.json',dict(result,source_members=manifest))
    return result


def verify(ctx,args):
    # Regression verification is part of publication QA, including the protected
    # final window; it is not a new scientific assignment/training campaign.
    from .cli import unit_tests
    tests=unit_tests(ctx)
    qa,isolated=render_and_isolate(ctx,args)
    from .csv_audit import run as audit
    arithmetic=audit(ctx,qa)
    pdf=compile_and_render(ctx,qa)
    review=v5_review_copy(ctx,qa)
    manifest=[]
    for asset in ASSETS:
        for p in sorted((ctx.root/'publication_bundle/numerical_data'/asset).iterdir()):
            if p.is_file():manifest.append(dict(asset=asset,relative_path='numerical_data/'+asset+'/'+p.name,
                size_bytes=p.stat().st_size,sha256=sha(p),isolated_reproduction='PASS'))
    pd.DataFrame(manifest).to_csv(ctx.root/'publication_bundle/figure_manifest.csv',index=False)
    from .publication_data import refresh_generated_index
    refresh_generated_index(ctx.root/'publication_bundle/numerical_data',ASSETS)
    result=dict(status='AUTOMATED_CHECKS_PASS_VISUAL_QA_PENDING',assets=len(isolated),tests=tests,arithmetic=arithmetic,pdf=pdf,v5_review_copy=review,qa_root=str(qa))
    write_json(ctx.root/'publication_bundle/AUTOMATED_VERIFICATION.json',result)
    return result
