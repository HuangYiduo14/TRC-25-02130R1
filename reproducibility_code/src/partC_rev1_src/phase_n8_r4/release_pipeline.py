"""Explicit extraction, portable rendering and file-access isolation verification."""
from pathlib import Path
import ast
import json
import os
import shutil
import subprocess
import sys
import time
import re
import numpy as np
import pandas as pd
from partC_rev1_src.phase_n8.common import read_json,write_json,sha


def build(ctx,args):
    from .release_data import Release,scenario_evidence,figures_3_6_7,figures_4_5_S1,figure_S2,figure_S3,figure_S4,figure_8
    from .release_tables import build as tables
    from .release_manuscript import write as manuscript
    release=Release(ctx);scenarios,curves=scenario_evidence(ctx)
    scenarios.to_csv(ctx.root/'tables/SCENARIO_RESULTS.csv',index=False,float_format='%.17g')
    curves.to_csv(ctx.root/'tables/FRESH_GAP_HISTORY.csv',index=False,float_format='%.17g')
    for operation in (lambda:figures_3_6_7(release,scenarios,curves),lambda:figures_4_5_S1(release),lambda:figure_S2(release),
        lambda:figure_S3(release),lambda:figure_S4(release),lambda:figure_8(release,scenarios)):
        ctx.guard();operation()
    release.finish_figures();ctx.guard();facts=tables(release,scenarios);text=manuscript(release,facts)
    inventory=[];claims=[]
    for name,specs in release.specs.items():
        if not name.startswith('fig_'):continue
        folder=release.root/name
        for spec in specs:
            files=spec['data'].split('|')+([spec['overlay']] if spec.get('overlay') else [])
            inventory.append(dict(figure=name,panel=spec['panel'],caption=spec['title'],status='EXTRACTED_AWAITING_RENDER',
                included_by_TeX=True,data_files='|'.join(files),code='numerical_release/'+name+'/plot.py',
                code_sha256=sha(folder/'plot.py'),pdf='numerical_release/'+name+'/'+spec['panel']+'.pdf',png='numerical_release/'+name+'/'+spec['panel']+'.png'))
            claims.append(dict(claim_id=name+'_'+spec['panel'],claim_scope='FIGURE_PANEL_ALL_PLOTTED_SERIES',
                csv='|'.join('numerical_release/'+name+'/'+f for f in files),row_selector='ALL supplied observations; grouping and display exclusions in plots.csv',
                value=spec['title'],unit=spec.get('xlabel','')+' / '+spec.get('ylabel',''),
                denominator=release.notes.get(name,''),source_locator='numerical_release/'+name+'/plots.csv / '+spec['panel']))
    pd.DataFrame(inventory).to_csv(ctx.root/'figure_inventory.csv',index=False)
    existing=pd.read_csv(ctx.root/'claim_to_result.csv');pd.concat([existing,pd.DataFrame(claims)],ignore_index=True).to_csv(ctx.root/'claim_to_result.csv',index=False,float_format='%.17g')
    pd.DataFrame([dict(table='table_'+str(i),status='CSV_AND_CODE_READY_AWAITING_BUILD',
        csv='numerical_release/table_'+str(i)+'/table.csv',code='numerical_release/table_'+str(i)+'/build_table.py',
        TeX='numerical_release/table_'+str(i)+'/table.tex') for i in range(1,6)]).to_csv(ctx.root/'table_inventory.csv',index=False)
    (release.root/'REPRODUCE.md').write_text('# Reproduce the numerical release\n\nEach figure directory is independent: run `python plot.py` with Python, NumPy, pandas and Matplotlib. Each table builds with `python build_table.py` using only the standard library. Scripts resolve CSVs beside themselves and may run from any current directory.\n\nCompile `main_numerical.tex` after drawing figures and building tables. The isolated portable compiler and exact commands are recorded outside this release in the run receipts. No original scientific inputs, repository imports, model inference or network calls are required for drawing.\n\nAll plot-source CSVs, including distribution observations and true map coordinates, are included. Native large scientific inputs remain authenticated in source_index.csv. The release is not the complete national raw database. See each dictionary for units, weights, missing coverage and graphic sampling.\n',encoding='utf-8')
    (release.root/'ACCESS_AND_LICENSE_NOTES.md').write_text('# Access and license scope\n\nThis is a local review artifact. No public upload or third-party redistribution approval is asserted. Plot CSVs are extracted analysis-ready observations and true source geometry used in the reported figures. Underlying FAF, county-factor, AADT, WIM and other source permissions remain attached to those sources. Unqualified independent-observation candidates are not represented as validation data. No fake public repository URL is provided. The portable Tectonic executable and downloaded TeX cache are not bundled as project data.\n',encoding='utf-8')
    extraction=release.root/'extraction_code';extraction.mkdir(exist_ok=True);extract_manifest=[]
    for name in ('release_data.py','release_tables.py','release_manuscript.py','release_pipeline.py',
                 'physical_population.py','path_metrics.py','regional_metrics.py','stability.py',
                 'geography.py','regional_geography.py','boundary_diagnostics.py','accounting.py'):
        source=Path(__file__).with_name(name);shutil.copy2(source,extraction/name)
        extract_manifest.append(dict(file=name,source_path=str(source),sha256=sha(source)))
    pd.DataFrame(extract_manifest).to_csv(extraction/'SOURCE_MANIFEST.csv',index=False)
    (extraction/'README.md').write_text('# Extraction layer\n\nThese exact source files generate the analysis-ready CSVs from the authenticated local project and native outputs. They are audit copies of the phase_n8_r4 package and retain its relative/project imports. The complete experiment package and transitive local dependency snapshot are supplied separately in the review handoff. They are not required by any plot.py.\n\nEvery figure is reproduced independently from its own CSVs using its local plot.py. Reopening native science/extraction uses an explicit authorized phase/run/configuration and exact indexed native inputs; this archive does not authorize restarting the expired campaign or overwriting its outputs.\n',encoding='utf-8')
    write_json(release.root/'BUILD_EXTRACTION_RESULT.json',dict(text,status='EXTRACTED',figure_folders=10,panels=len(inventory)))
    return dict(status='EXECUTED',panels=len(inventory),tables=5,publication_assets_complete=False)


def scan_plot(path):
    tree=ast.parse(path.read_text(encoding='utf-8'));imports=[];bad=[]
    allowed={'pathlib','csv','hashlib','textwrap','numpy','pandas','matplotlib'}
    forbidden={'eval','exec','__import__','compile','subprocess','importlib','pickle','socket','requests','urllib','geopandas','PIL'}
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):imports.extend(a.name for a in node.names)
        elif isinstance(node,ast.ImportFrom):imports.append(node.module or '')
        if isinstance(node,ast.Name) and node.id in forbidden:bad.append(node.id)
        if isinstance(node,ast.Constant) and isinstance(node.value,str) and any(x in node.value.lower() for x in ('http://','https://','.npz','.parquet','.pkl','.shp')):bad.append(node.value)
    if any(x.split('.')[0] not in allowed for x in imports) or bad:raise ValueError('Non-independent drawing source: '+repr((imports,bad)))
    return dict(imports=imports,forbidden_constructs=bad,manual_file_opening_policy='read() permits only immediate local CSV paths; plot.py itself and standard library/package/font/cache reads are audited separately')


AUDIT_DRIVER=r'''
import sys,os,json,runpy
from pathlib import Path
root=Path(sys.argv[1]).resolve();repo=Path(sys.argv[2]).resolve();events=[]
prefix=Path(sys.prefix).resolve();base=Path(sys.base_prefix).resolve()
system=Path(os.environ.get('WINDIR','C:/Windows')).resolve()
def inside(p,r):return p==r or r in p.parents
def audit(event,args):
    if event!='open' or not args or isinstance(args[0],int):return
    try:p=Path(args[0]).resolve()
    except (TypeError,ValueError):return
    mode=args[1] if len(args)>1 else None
    writing=isinstance(mode,str) and any(c in mode for c in 'wax+')
    if inside(p,repo):raise RuntimeError('Repository file access prohibited: '+str(p))
    if writing:
        if not inside(p,root):raise RuntimeError('Drawing write outside isolated folder: '+str(p))
        return
    if inside(p,root):
        if p==root/'plot.py' or inside(p,root/'.mplcache'):return
        if p.parent==root and p.suffix=='.csv':events.append(str(p.relative_to(root)));return
        raise RuntimeError('Non-CSV drawing input: '+str(p))
    if inside(p,prefix) or inside(p,base) or inside(p,system):return
    raise RuntimeError('External drawing-data access: '+str(p))
sys.addaudithook(audit)
runpy.run_path(str(root/'plot.py'),run_name='__main__')
(root/'runtime_file_access.json').write_text(json.dumps({'csv_reads':sorted(set(events)),'repository_access':False,'non_CSV_data_inputs':False},indent=2))
'''


def render(ctx,args):
    release=ctx.root/'numerical_release';qa=ctx.root/'validation/figure_isolation'/args.job_id;qa.mkdir(parents=True,exist_ok=False)
    required_figures={'fig_'+str(i) for i in range(3,9)}|{'fig_S'+str(i) for i in range(1,5)}
    required_tables={'table_'+str(i) for i in range(1,6)}
    actual_figures={p.name for p in release.glob('fig_*') if p.is_dir()}
    actual_tables={p.name for p in release.glob('table_*') if p.is_dir()}
    if actual_figures!=required_figures or actual_tables!=required_tables:
        raise ValueError('Required release directories differ: '+repr((actual_figures,actual_tables)))
    # Empty PYTHONPATH and isolated interpreter; cwd is never the repository.
    env=dict(os.environ,PYTHONPATH='',PYTHONDONTWRITEBYTECODE='1',MPLBACKEND='Agg',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    records=[]
    for folder in sorted(release.glob('fig_*')):
        if not folder.is_dir():continue
        ctx.guard();scan=scan_plot(folder/'plot.py');baseline_log=qa/(folder.name+'_baseline.log')
        localenv=dict(env,MPLCONFIGDIR=str(folder/'.mplcache'))
        with baseline_log.open('w',encoding='utf-8') as log:
            done=subprocess.run([sys.executable,'-I','-B',str(folder/'plot.py')],cwd=qa,env=localenv,stdout=log,stderr=subprocess.STDOUT,timeout=min(900,max(1,ctx.deadline-time.time()-30)))
        if done.returncode:raise RuntimeError('Baseline independent figure render failed: '+str(baseline_log))
        panels=pd.read_csv(folder/'plots.csv',keep_default_na=False)
        if panels.empty or panels.panel.duplicated().any():raise ValueError('Missing or duplicated panel inventory: '+folder.name)
        for panel in panels.panel:
            for suffix in ('.pdf','.png'):
                output=folder/(panel+suffix)
                if not output.is_file() or output.stat().st_size==0:raise ValueError('Required panel output missing: '+str(output))
        isolated=qa/(folder.name+'_ALONE')
        if isolated.exists():raise FileExistsError('Isolation location must be new: '+str(isolated))
        shutil.copytree(folder,isolated,ignore=shutil.ignore_patterns('.mplcache','*.pdf','*.png',
            'rendered_series_audit.csv','read_csv_hashes.csv','FILE_HASHES.csv'))
        ilog=qa/(folder.name+'_isolated.log');localenv=dict(env,MPLCONFIGDIR=str(isolated/'.mplcache'))
        command=[sys.executable,'-I','-B','-c',AUDIT_DRIVER,str(isolated),str(ctx.repo)]
        with ilog.open('w',encoding='utf-8') as log:
            done=subprocess.run(command,cwd=qa,env=localenv,stdout=log,stderr=subprocess.STDOUT,timeout=min(900,max(1,ctx.deadline-time.time()-30)))
        if done.returncode:raise RuntimeError('Isolated CSV-only render failed: '+str(ilog))
        for panel in panels.panel:
            for suffix in ('.pdf','.png'):
                output=isolated/(panel+suffix)
                if not output.is_file() or output.stat().st_size==0:
                    raise ValueError('Isolated run did not regenerate required panel: '+str(output))
        for name in ('rendered_series_audit.csv','read_csv_hashes.csv'):
            if sha(folder/name)!=sha(isolated/name):raise ValueError('Isolated drawing numerical/hash audit differs: '+folder.name+'/'+name)
        # Exact PNG pixels are a stronger rendering comparison than PDF timestamp bytes.
        import matplotlib.image as mpimg
        for p in folder.glob('panel_*.png'):
            if not np.array_equal(mpimg.imread(p),mpimg.imread(isolated/p.name)):raise ValueError('Isolated pixels differ: '+str(p))
        item=dict(figure=folder.name,status='PASS',source_plot_sha256=sha(folder/'plot.py'),
            baseline_command=[sys.executable,'-I','-B',str(folder/'plot.py')],isolated_command=command,
            cwd=str(qa),empty_PYTHONPATH=True,repository_access_denied=True,non_CSV_drawing_input_denied=True,
            preexisting_PDF_PNG_and_derived_audits_excluded=True,
            series_and_input_hashes_identical=True,png_pixels_identical=True,AST=scan,
            actual_csv_reads=read_json(isolated/'runtime_file_access.json')['csv_reads'])
        records.append(item);write_json(qa/'ISOLATION_RESULT.json',dict(status='IN_PROGRESS',figures=records))
        print('FIGURE_ISOLATION_PASS',folder.name,flush=True)
    built_tables=[]
    for folder in sorted(release.glob('table_*')):
        if not folder.is_dir():continue
        ctx.guard();logpath=qa/(folder.name+'_build.log')
        with logpath.open('w',encoding='utf-8') as log:
            done=subprocess.run([sys.executable,'-I','-B',str(folder/'build_table.py')],cwd=qa,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=60)
        if done.returncode:raise RuntimeError('Standalone table build failed: '+str(logpath))
        if not (folder/'table.tex').is_file() or not (folder/'table.tex').stat().st_size:raise ValueError('Missing built table: '+folder.name)
        built_tables.append(folder.name)
    if {r['figure'] for r in records}!=required_figures or set(built_tables)!=required_tables:
        raise ValueError('Incomplete standalone validation')
    figure_inventory=pd.read_csv(ctx.root/'figure_inventory.csv',keep_default_na=False)
    for i,row in figure_inventory.iterrows():
        folder=release/row.figure
        figure_inventory.loc[i,'status']='ISOLATION_PASS_PENDING_VISUAL_QA'
        figure_inventory.loc[i,'code_sha256']=sha(folder/'plot.py')
        figure_inventory.loc[i,'pdf_sha256']=sha(ctx.root/row.pdf)
        figure_inventory.loc[i,'png_sha256']=sha(ctx.root/row.png)
        figure_inventory.loc[i,'data_sha256']=json.dumps({name:sha(folder/name) for name in row.data_files.split('|')})
    figure_inventory.to_csv(ctx.root/'figure_inventory.csv',index=False)
    table_inventory=pd.read_csv(ctx.root/'table_inventory.csv',keep_default_na=False)
    for i,row in table_inventory.iterrows():
        table_inventory.loc[i,'status']='BUILT_PENDING_PDF_VISUAL_QA'
        for field in ('csv','code','TeX'):table_inventory.loc[i,field+'_sha256']=sha(ctx.root/row[field])
    table_inventory.to_csv(ctx.root/'table_inventory.csv',index=False)
    for name in sorted(required_figures|required_tables):
        folder=release/name
        local=[dict(file=p.name,size_bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(folder.iterdir())
            if p.is_file() and p.name!='FILE_HASHES.csv']
        pd.DataFrame(local).to_csv(folder/'FILE_HASHES.csv',index=False)
    bindings=[]
    for name in sorted(required_figures|required_tables):
        for p in sorted((release/name).iterdir()):
            if p.is_file():bindings.append(dict(relative_path=p.relative_to(release).as_posix(),sha256=sha(p),size_bytes=p.stat().st_size))
    write_json(qa/'ISOLATION_RESULT.json',dict(status='PASS',completed_epoch=time.time(),figures=records,
        all_ten_figures=True,standalone_tables=built_tables,current_release_bindings=bindings))
    return dict(status='PASS',figures=len(records),standalone_tables=len(built_tables),visual_inspection_pending=True)


def compile_pdf(ctx,args):
    release=ctx.root/'numerical_release';qa=ctx.root/'validation/pdf_render'/args.job_id;qa.mkdir(parents=True,exist_ok=False)
    toolchain=ctx.root/'toolchain';exe=toolchain/'tectonic.exe'
    ctx.bind(toolchain/'tectonic.exe','USER_APPROVED_ISOLATED_PORTABLE_LATEX_COMPILER')
    env=dict(os.environ,TECTONIC_CACHE_DIR=str(toolchain/'cache'),XDG_CACHE_HOME=str(toolchain/'cache'),
        XDG_CONFIG_HOME=str(toolchain/'config'))
    command=[str(exe),'--keep-logs','--keep-intermediates','--outdir',str(release),str(release/'main_numerical.tex')]
    logpath=qa/'tectonic_stdout_stderr.log'
    with logpath.open('w',encoding='utf-8') as log:
        done=subprocess.run(command,cwd=release,env=env,stdout=log,stderr=subprocess.STDOUT,
            timeout=min(1200,max(1,ctx.deadline-time.time()-120)))
    if done.returncode:raise RuntimeError('Portable manuscript compilation failed: '+str(logpath))
    pdf=release/'main_numerical.pdf'
    if not pdf.exists() or pdf.stat().st_size==0:raise ValueError('Compiler returned no PDF')
    ctx.guard();poppler=Path(os.environ['USERPROFILE'])/'.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin'
    textpath=qa/'main_numerical.txt'
    # This bundled Poppler build supplies pdftoppm but not pdftotext. Use the
    # existing isolated artifact runtime for text extraction without modifying
    # the scientific Python environment or installing any package.
    artifact_python=poppler.parents[3]/'python/python.exe'
    text_command=[str(artifact_python),'-I','-B','-c',
        "from pathlib import Path; from pypdf import PdfReader; import sys; "
        "r=PdfReader(sys.argv[1]); Path(sys.argv[2]).write_text('\\n\\f\\n'.join(p.extract_text(extraction_mode='layout') or '' for p in r.pages),encoding='utf-8')",
        str(pdf),str(textpath)]
    subprocess.run(text_command,check=True,timeout=120)
    extracted=textpath.read_text(encoding='utf-8')
    if '@@' in extracted or '[?]' in extracted or '??' in extracted:raise ValueError('Unresolved manuscript fact/citation/reference in compiled text')
    source_log=(release/'main_numerical.log').read_text(encoding='utf-8',errors='replace')
    issues=[line for line in source_log.splitlines() if re.search(r'Overfull|undefined|not found|Missing character',line,re.I)]
    (qa/'layout_and_reference_warnings.txt').write_text('\n'.join(issues)+'\n',encoding='utf-8')
    render_command=[str(poppler/'pdftoppm.exe'),'-scale-to','1400','-png',str(pdf),str(qa/'page')]
    subprocess.run(render_command,check=True,timeout=min(900,max(1,ctx.deadline-time.time()-30)))
    pages=sorted(qa.glob('page-*.png'))
    if not pages:raise ValueError('PDF produced no rendered pages')
    # Contact sheets are QA aids only; drawing folders never read raster backgrounds.
    from PIL import Image,ImageOps,ImageDraw
    sheets=[]
    for begin in range(0,len(pages),4):
        canvas=Image.new('RGB',(1800,2460),'#dddddd');draw=ImageDraw.Draw(canvas)
        for i,p in enumerate(pages[begin:begin+4]):
            with Image.open(p) as im:tile=ImageOps.contain(im.convert('RGB'),(880,1190))
            xx=(i%2)*900+(900-tile.width)//2;yy=(i//2)*1230+25
            canvas.paste(tile,(xx,yy));draw.text(((i%2)*900+12,(i//2)*1230+8),p.name,fill='black')
        out=qa/f'contact_{begin//4+1:03d}.png';canvas.save(out);sheets.append(str(out))
    manuscript_inputs=set(release.glob('*.tex'))|set(release.glob('*.bib'))|{release/'numerical_facts.csv'}
    manuscript_inputs|=set(release.glob('fig_*/panel_*.pdf'))|set(release.glob('table_*/table.tex'))
    manuscript_bindings=[dict(relative_path=p.relative_to(release).as_posix(),sha256=sha(p)) for p in sorted(manuscript_inputs)]
    result=dict(status='COMPILED_AND_RENDERED_PENDING_VISUAL_QA',command=command,pdf=str(pdf),pdf_sha256=sha(pdf),
        manuscript_input_bindings=manuscript_bindings,
        pages=len(pages),text_extraction_command=text_command,render_command=render_command,rendered_pages=[str(p) for p in pages],contact_sheets=sheets,
        layout_reference_warning_lines=issues,user_approved_portable_toolchain=True,source_python_environments_changed=False)
    write_json(qa/'PDF_BUILD_RESULT.json',result);write_json(ctx.root/'validation/CURRENT_PDF_BUILD.json',result)
    return dict(status='EXECUTED',pdf_pages=len(pages),layout_warning_lines=len(issues),visual_inspection_pending=True)
