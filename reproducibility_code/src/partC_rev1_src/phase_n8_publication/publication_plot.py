"""Standalone drawing from the CSV files beside this file."""
from pathlib import Path
import csv
import hashlib
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize,TwoSlopeNorm,LogNorm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator

ROOT=Path(__file__).resolve().parent
READS={}
ASSET=''
STEMS={'figure4':{'geometry':'road_vertices'},'figure5':{'observations':'region_fraction_observations'},
    'figure6':{'active_dimensions':'dimensions'},'figure7':{'backbone_flows':'flow_open_closed','mask_scope':'source_coverage'},
    'figure8':{'components':'cost_components','query_costs':'cumulative_costs','offline_scopes':'comparison_scope'},
    'figureS1':{'batch_losses':'optimizer_observations'},'figureS2':{'ownership':'membership'},
    'figureS3':{'fallback':'regional_fallback','safety_queries':'safety_diagnostics'},'figureS4':{'backbone_flows':'flow_open_closed'}}
COLORS=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#666666']
METHOD_COLORS={'M1':COLORS[0],'M2':COLORS[1],'M3':COLORS[2]}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,
    'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,
    'savefig.dpi':180,'figure.constrained_layout.use':True})


def read(stem):
    stem=STEMS.get(ASSET,{}).get(stem,stem)
    files=[ROOT/(stem+'.csv')] if (ROOT/(stem+'.csv')).exists() else sorted(ROOT.glob(stem+'_*.csv'))
    if not files:raise FileNotFoundError('Required local CSV: '+stem)
    parts=[]
    for path in files:
        if path.parent!=ROOT or path.suffix!='.csv':raise ValueError('Only local CSV data permitted')
        frame=pd.read_csv(path,float_precision='round_trip',keep_default_na=True,dtype={
            'abstract_link_id':str,'feature_id':str,'physical_arc_id':str,'region_id':str,'input_hash':str,'physical_node_uid':str})
        READS[path.name]=(hashlib.sha256(path.read_bytes()).hexdigest(),len(frame));parts.append(frame)
    return pd.concat(parts,ignore_index=True)


def label(ax,title,x=None,y=None):
    ax.set_title(title,loc='left',fontweight='bold')
    if x is not None:ax.set_xlabel(x)
    if y is not None:ax.set_ylabel(y)
    ax.grid(alpha=.15,linewidth=.4)


def note(ax,text,width=80):
    ax.axis('off');ax.text(.02,.88,textwrap.fill(text,width),va='top',transform=ax.transAxes,fontsize=9)


def readable_symlog_ticks(ax):
    lower,upper=ax.get_ylim()
    candidates=np.r_[0.,(10.**np.arange(-6,10)[:,None]*np.array([1.,2.,5.])[None,:]).ravel()]
    ticks=candidates[(candidates>=max(0,lower))&(candidates<=upper)]
    if len(ticks)>7:
        transform=ax.yaxis.get_transform();position=transform.transform(ticks)
        targets=np.linspace(position[0],position[-1],7)
        ticks=ticks[np.unique([int(np.argmin(abs(position-t))) for t in targets])]
    ax.set_yticks(ticks,[format(v,'.3g') for v in ticks])


def finite(frame,cols):
    ok=np.isfinite(frame[cols].to_numpy(float)).all(axis=1)
    return frame.loc[ok]


def segments(frame,idcol='feature_id'):
    order=[idcol,'part_id','vertex_order'];f=frame.sort_values(order)
    same=(f[idcol].to_numpy()[1:]==f[idcol].to_numpy()[:-1])&(f.part_id.to_numpy()[1:]==f.part_id.to_numpy()[:-1])
    xy=f[['x','y']].to_numpy(float)/1000
    return np.stack([xy[:-1][same],xy[1:][same]],axis=1),f.iloc[:-1].loc[same]


def map_lines(ax,geometry,values=None,*,signed=False,vmax=None,cmap=None,width=.3,extent=None,bar_label=''):
    seg,start=segments(geometry)
    if values is None:
        coll=LineCollection(seg,colors='#a7adb4',linewidths=width,rasterized=True)
    else:
        val=start.abstract_link_id.map(values).to_numpy(float)
        if not np.isfinite(val).all():raise ValueError('Map value/geometry join incomplete')
        v=float(np.max(abs(val))) if vmax is None else vmax
        if v<=0:v=1.
        norm=TwoSlopeNorm(vmin=-v,vcenter=0,vmax=v) if signed else Normalize(0,v)
        coll=LineCollection(seg,cmap=cmap or ('RdBu_r' if signed else 'viridis'),norm=norm,linewidths=width,rasterized=True)
        coll.set_array(val)
    ax.add_collection(coll);ax.autoscale();ax.set_aspect('equal')
    if extent:ax.set_xlim(extent[:2]);ax.set_ylim(extent[2:])
    ax.set_xlabel('Easting (km, EPSG:5070)');ax.set_ylabel('Northing (km)')
    if values is not None:
        cb=plt.colorbar(coll,ax=ax,shrink=.75,pad=.01);cb.set_label(bar_label,fontsize=7);cb.ax.tick_params(labelsize=7)
    return coll


def grouped_lines(ax,f,x,y,group,logx=False,logy=False):
    for i,(name,g) in enumerate(f.groupby(group,sort=False)):
        g=finite(g,[x,y]).sort_values(x)
        if len(g):
            color=METHOD_COLORS.get(str(g.iloc[0].get('method_id','')),COLORS[i%len(COLORS)])
            recovery=str(g.iloc[0].get('assignment_role',''))=='RECOVERY_CAMPAIGN'
            ax.plot(g[x],g[y],label=str(name),color=color,ls='--' if recovery else '-',alpha=.5 if recovery else 1,marker='.' if recovery else 'o',ms=3,lw=1)
    if logx:ax.set_xscale('log')
    if logy:ax.set_yscale('log')
    ax.legend(frameon=False)


def main_figure(asset):
    if asset=='figure3':
        ref=read('reference');per=read('per_link');keys=read('key_links');geo=read('geometry');history=read('convergence')
        fig,axs=plt.subplots(3,2,figsize=(12,13));a=axs.ravel()
        map_lines(a[0],geo,width=.12)
        a[0].scatter(keys.x/1000,keys.y/1000,s=12,c='#D55E00',edgecolors='white',linewidths=.2,zorder=3)
        label(a[0],'(a) Frozen 200 high-flow abstract links')
        a[1].hist(ref.lambda_reference,bins=np.r_[0,np.geomspace(max(1e-10,ref.loc[ref.lambda_reference>0,'lambda_reference'].min()),ref.lambda_reference.max()*1.00001,70)],color='#547d91')
        a[1].set_xscale('symlog',linthresh=1);a[1].set_yscale('log');label(a[1],'(b) Complete backbone reference','Annual kilotons','Link count')
        a[1].text(.03,.96,'Zero-flow links: '+format(int(ref.lambda_reference.eq(0).sum()),',')+'\nIncluded in full-population metrics',va='top',transform=a[1].transAxes,fontsize=8,bbox=dict(facecolor='white',alpha=.9,edgecolor='none'))
        for j,(take,title) in enumerate([(per,'(c) Complete backbone errors'),(per.loc[per.is_key_link],'(d) Fixed 200-link errors')],2):
            for i,(method,g) in enumerate(take.groupby('method_label',sort=False)):
                bins=np.linspace(-1,1,61);a[j].hist(g.signed_bounded_error,bins=bins,histtype='step',density=True,color=METHOD_COLORS[str(g.iloc[0].method_id)],label=method,lw=1.3)
            a[j].legend(frameon=False,fontsize=7);label(a[j],title,'Signed bounded error','Density');a[j].set_yscale('symlog',linthresh=.1)
        for i,(method,g) in enumerate(per.loc[per.is_key_link].groupby('method_label',sort=False)):
            a[4].scatter(g.lambda_reference,g.lambda_estimate,s=12,alpha=.6,label=method,color=METHOD_COLORS[str(g.iloc[0].method_id)],rasterized=True)
        maximum=per.loc[per.is_key_link,['lambda_reference','lambda_estimate']].max().max()
        a[4].plot([0,maximum],[0,maximum],color='black',ls='--',lw=.8);a[4].set_xscale('symlog',linthresh=1);a[4].set_yscale('symlog',linthresh=1)
        a[4].legend(frameon=False,fontsize=7);label(a[4],'(e) Fixed-link flows','Reference annual kilotons','Predicted annual kilotons')
        grouped_lines(a[5],history,'elapsed_seconds','RG_SPTT','curve_label',logy=True);a[5].axhline(.001,c='black',ls=':',lw=.8)
        label(a[5],'(f) Fresh certificates by fixed objective','Epoch elapsed (s)','Relative gap')
    elif asset=='figure4':
        roads=read('geometry');points=read('terminals');locations=read('locations');background=read('national_geometry')
        fig,axs=plt.subplots(2,2,figsize=(12,10));a=axs.ravel();map_lines(a[0],background,width=.15)
        for row in locations.itertuples():
            a[0].scatter(row.x/1000,row.y/1000,s=40,zorder=3);a[0].annotate(row.role_label,(row.x/1000,row.y/1000),xytext=(4,4),textcoords='offset points',fontsize=8)
        label(a[0],'(a) Representative regional locations')
        for ax,row,letter in zip(list(a)[1:],locations.itertuples(),'bcd'):
            map_lines(ax,roads.loc[roads.region_id.eq(row.region_id)],width=.55)
            p=points.loc[points.region_id.eq(row.region_id)]
            styles={'BACN':('o','#0072B2','B-ACN portal'),'RACN':('s','#009E73','R-ACN portal'),
                'ORIGIN_CENTROID':('^','#D55E00','Origin centroid'),'DESTINATION_CENTROID':('v','#CC79A7','Destination centroid')}
            for name,g in p.groupby('terminal_type'):
                marker,color,title=styles[name];ax.scatter(g.x/1000,g.y/1000,s=28,marker=marker,
                    color=color,edgecolors='white',linewidths=.3,label=title,zorder=4)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4));ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.legend(frameon=False,fontsize=7);label(ax,'('+letter+') '+row.role_label)
    elif asset=='figure5':
        data=read('observations');coverage=read('label_coverage');fits=read('fit_coverage')
        fig,axs=plt.subplots(2,2,figsize=(12,9))
        strata=list(fits.stratum.drop_duplicates())
        for ax,stratum,letter in zip(axs.flat,strata,'abcd'):
            f=data.loc[data.stratum.eq(stratum)&data.role.eq('Fresh operating')&np.isfinite(data.cost_wmape_percent)]
            for i,(family,g) in enumerate(f.groupby('model_family',sort=False)):
                summary=g.groupby('fraction').cost_wmape_percent.agg(median='median',q1=lambda x:x.quantile(.25),q3=lambda x:x.quantile(.75))
                counts=g.groupby('fraction').region_id.nunique().reindex([.1,.25,.5,.7,1.],fill_value=0)
                ax.plot(summary.index*100,summary['median'],marker='o',color=COLORS[i],label=family+' (n='+','.join(str(x) for x in counts)+')')
                ax.fill_between(summary.index.to_numpy()*100,summary.q1,summary.q3,color=COLORS[i],alpha=.16)
            ax.set_xticks([10,25,50,70,100]);lower,upper=ax.get_ylim()
            if lower>0 and upper/lower<50:
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5));ax.ticklabel_format(axis='y',style='plain',useOffset=False);scale='linear'
            else:
                ax.set_yscale('symlog',linthresh=.01);ax.set_ylim(bottom=0);readable_symlog_ticks(ax);scale='symlog'
            label(ax,'('+letter+') '+str(stratum).replace('_',' ').title(),'TRAIN fraction (%)','Cost WMAPE (%; '+scale+')')
            if len(f):ax.legend(frameon=False,fontsize=7)
            else:ax.text(.5,.5,'No admitted cost-WMAPE observations',ha='center',va='center',transform=ax.transAxes)
        fig.suptitle('Median and interquartile range across regions; seed 0\nPost-fit operating labels: eight regional datasets reused unchanged\nLegend n: finite regions at 10, 25, 50, 70, 100% TRAIN; exclusions and stress results in local CSVs',fontsize=10)
    elif asset=='figure6':
        f=read('suite_metrics').sort_values('resolution');dims=read('active_dimensions');fig,axs=plt.subplots(2,3,figsize=(14,8));a=axs.ravel()
        for col,title in [('backbone_count','Backbone links'),('active_regions','Active regions')]:a[0].plot(f.resolution,f[col],marker='o',label=title)
        a[0].set_yscale('log');a[0].legend(frameon=False);label(a[0],'(a) Represented graph size','Resolution','Count')
        for ax,col,title,y in zip(list(a)[1:5],['online_seconds','backbone_wmape','key_wmape','backbone_cost_error'],['(b) Complete assignment time','(c) Complete backbone error','(d) Fixed-key error','(e) Backbone cost error'],['Seconds','WMAPE (%)','WMAPE (%)','Absolute error (%)']):
            ax.plot(f.resolution,f[col],'-o',color=COLORS[0]);label(ax,title,'Resolution',y)
            ax.ticklabel_format(axis='y',style='plain',useOffset=False)
        resolution=f.set_index('suite_id').resolution.to_dict()
        for sid,g in dims.groupby('suite_id',sort=False):
            vals=np.sort(g.m_active.to_numpy());a[5].plot(vals,np.arange(1,len(vals)+1)/len(vals),label='Resolution '+format(resolution[sid],'.4g'))
        a[5].set_xscale('log');a[5].legend(frameon=False);label(a[5],'(f) Active regional dimensions','Active pairs per region','Cumulative region fraction')
        for ax in list(a)[:5]:ax.set_xscale('log');ax.set_xticks(f.resolution,labels=[format(x,'.4g') for x in f.resolution]);ax.tick_params(axis='x',labelsize=7)
    elif asset in ('figure7','figureS4'):
        b=read('backbone_flows');r=read('regional_flows');geo=read('geometry');rgeo=read('regional_geometry');scope=read('mask_scope')
        methods=list(b.method_label.drop_duplicates());rows=len(methods);fig,axs=plt.subplots(rows,4,figsize=(16,4*rows),squeeze=False)
        extent=[-850,2300,200,3300]
        for axes,method in zip(axs,methods):
            bb=b.loc[b.method_label.eq(method)];rr=r.loc[r.method_label.eq(method)];bg=geo.loc[geo.abstract_link_id.isin(bb.abstract_link_id)];rg=rgeo.loc[rgeo.abstract_link_id.isin(rr.abstract_link_id)]
            mask=bb.set_index('abstract_link_id').is_closed.astype(float)
            map_lines(axes[0],bg,width=.12,extent=extent)
            closed_geo=bg.loc[bg.abstract_link_id.isin(bb.loc[bb.is_closed,'abstract_link_id'])]
            closed_segments,_=segments(closed_geo)
            axes[0].add_collection(LineCollection(closed_segments,colors='#B2182B',linewidths=1.1,rasterized=True))
            axes[0].plot([],[],color='#B2182B',lw=2,label='Closed represented backbone');axes[0].legend(frameon=False,fontsize=7,loc='lower right')
            vmax=float(bb[['flow_open','flow_closed']].max().max());rvmax=float(rr[['flow_open','flow_closed']].max().max())
            for ax,col,title in zip(axes[1:3],['flow_open','flow_closed'],['Open','Represented closure']):
                map_lines(ax,rg,rr.set_index('abstract_link_id')[col],vmax=rvmax,width=.15,extent=extent,cmap='Greys',bar_label='Regional chords (annual kt)')
                map_lines(ax,bg,bb.set_index('abstract_link_id')[col],vmax=vmax,width=.35,extent=extent,bar_label='Backbone (annual kt)');label(ax,title)
            map_lines(axes[3],rg,rr.set_index('abstract_link_id').flow_delta,signed=True,width=.15,extent=extent,cmap='PRGn',bar_label='Regional change (annual kt)')
            map_lines(axes[3],bg,bb.set_index('abstract_link_id').flow_delta,signed=True,width=.45,extent=extent,bar_label='Backbone change (annual kt)');label(axes[3],'Gain (+) / loss (−)')
            label(axes[0],textwrap.fill(method,29)+'\nI-95 represented mask')
        fig.suptitle('Eastern United States view; complete national flows are in the local CSVs\nAnnual kilotons; separate backbone/regional scales; 1,105 represented source arcs closed and 608 interior arcs omitted',fontsize=11)
    elif asset=='figure8':
        comp=read('components');query=read('query_costs');summary=read('offline_scopes');fig=plt.figure(figsize=(12,8))
        grid=fig.add_gridspec(2,2,height_ratios=[3,1]);axs=[fig.add_subplot(grid[0,0]),fig.add_subplot(grid[0,1]),fig.add_subplot(grid[1,:])]
        selected=comp.loc[comp.display_in_composition]
        axs[0].barh([textwrap.fill(s,24) for s in selected.component_label],selected.seconds,color=COLORS[0]);label(axs[0],'(a) Recorded serial components','Seconds')
        axs[1].barh([textwrap.fill(s,25) for s in query.method_label],query.observed_job_seconds,color=COLORS[:len(query)])
        label(axs[1],'(b) Observed assignment job clocks','Seconds')
        note(axs[2],'Cold-start amortization is not comparable. New polynomial and final-mixture jobs reuse a separately timed preflight feasible state; historical FFTT and physical controls include fresh initialization. Offline records mix parent elapsed and worker wall-time sums and do not establish a full attributable intercept. No repeated-query crossing or equal-accuracy speedup is inferred.',width=125);axs[2].set_title('(c) Why no amortization curve is drawn',loc='left',fontweight='bold')
    elif asset=='figureS1':
        loss=read('batch_losses');valid=read('valid_checkpoints');counters=read('fit_counters');fig,axs=plt.subplots(1,3,figsize=(15,5))
        for i,(rid,g) in enumerate(loss.groupby('region_id',sort=False)):
            group=g.groupby('optimizer_updates').loss.median().reset_index() if 'optimizer_updates' in g else g
            x='optimizer_updates' if 'optimizer_updates' in group else 'update';y='loss' if 'loss' in group else 'loss_total'
            axs[0].plot(group[x],group[y],color=COLORS[i%6],alpha=.5,lw=.6)
        axs[0].set_yscale('symlog',linthresh=.01);label(axs[0],'(a) Fixed-data optimizer traces','Executed update','Robust training loss')
        axs[0].text(.02,.98,'One trace per region; colours repeat',va='top',transform=axs[0].transAxes,fontsize=7)
        for _,g in valid.groupby(['region_id','fraction'],sort=False):axs[1].plot(g['update'],g.VALID_MARE,color=COLORS[0],alpha=.55,lw=.8)
        axs[1].set_yscale('symlog',linthresh=.01);label(axs[1],'(b) Fixed VALID checkpoint selection','Update','VALID MARE (fraction)')
        counts=counters.groupby(['executed_updates','selected_updates']).size().reset_index(name='n')
        axs[2].scatter(counts.executed_updates,counts.selected_updates,color=COLORS[0],s=40+18*counts.n,alpha=.75)
        for row in counts.itertuples():axs[2].annotate('n='+str(row.n),(row.executed_updates,row.selected_updates),xytext=(12,0),textcoords='offset points',va='center',fontsize=8)
        axs[2].set_xlim(3800,4250)
        label(axs[2],'(c) Executed versus selected','Executed updates','Selected update')
        fig.suptitle(str(len(counters))+' regional fits at 100% TRAIN; 4,000 executed updates per fit',fontsize=11)
    elif asset=='figureS2':
        maps=read('ownership');stability=read('stability');counts=read('contingency');resolution=read('suite_resolutions').set_index('suite_id').resolution.to_dict();fig,axs=plt.subplots(2,3,figsize=(15,9))
        for ax,(sid,g),letter in zip(axs.flat,maps.groupby('suite_id',sort=False),'abcde'):
            c=pd.factorize(g.region_id,sort=True)[0]%20;ax.scatter(g.x/1000,g.y/1000,c=c,cmap='tab20',s=.15,rasterized=True)
            ax.set_aspect('equal');label(ax,'('+letter+') Resolution '+format(resolution[sid],'.4g')+' ownership','Easting (km, EPSG:5070)','Northing (km)')
        ids=list(maps.suite_id.drop_duplicates());matrix=np.eye(len(ids))
        for row in stability.itertuples():
            if row.suite_a in ids and row.suite_b in ids:matrix[ids.index(row.suite_a),ids.index(row.suite_b)]=row.ARI;matrix[ids.index(row.suite_b),ids.index(row.suite_a)]=row.ARI
        ax=axs.flat[5];im=ax.imshow(matrix,vmin=0,vmax=1,cmap='viridis');ticklabels=[format(resolution[sid],'.4g') for sid in ids];ax.set_xticks(range(5),ticklabels);ax.set_yticks(range(5),ticklabels)
        for i in range(5):
            for j in range(5):ax.text(j,i,format(matrix[i,j],'.3f'),ha='center',va='center',color='white' if matrix[i,j]<.6 else 'black',fontsize=8)
        plt.colorbar(im,ax=ax,shrink=.7);label(ax,'(f) Common-arc ownership ARI')
    elif asset=='figureS3':
        f=read('fallback');q=read('safety_queries');floor=read('floor_diagnostics');fig,axs=plt.subplots(2,2,figsize=(11,8));a=axs.ravel()
        for ax,value,title in [(a[0],None,'(a) Final families and replacements'),(a[1],'baseline_exposure','(b) Baseline traversal exposure')]:
            grouped=f.groupby(['effective_family','was_replaced']).size().unstack(fill_value=0) if value is None else f.groupby(['effective_family','was_replaced'])[value].sum().unstack(fill_value=0)
            left=np.zeros(len(grouped))
            for changed,color in [(False,COLORS[0]),(True,COLORS[1])]:
                values=grouped[changed].to_numpy() if changed in grouped else np.zeros(len(grouped))
                ax.barh(grouped.index,values,left=left,color=color,label='Replaced' if changed else 'Retained');left+=values
            ax.legend(frameon=False);label(ax,title,'Regions' if value is None else 'Annual kilotons')
        a[2].hist(q.minimum_cost_seconds.dropna(),bins=60,color=COLORS[0]);a[2].axvline(0,color='#D55E00',ls='--');a[2].set_xscale('symlog',linthresh=1);a[2].set_yscale('log')
        label(a[2],'(c) Saved and offending-query minima','Minimum cost (s)','Query / region count')
        total=floor.groupby('registry')[['negative_coordinates','below_fftt_coordinates']].sum()
        yy=np.arange(len(total));a[3].barh(yy-.17,total.below_fftt_coordinates,height=.32,label='Below FFTT',color=COLORS[0]);a[3].barh(yy+.17,total.negative_coordinates,height=.32,label='Negative',color=COLORS[1])
        a[3].set_yticks(yy,[textwrap.fill(s,22) for s in total.index]);a[3].set_xscale('symlog',linthresh=1);a[3].legend(frameon=False)
        label(a[3],'(d) Native fidelity diagnostics','Coordinate-query observations')
    else:raise ValueError('Unknown explicit figure '+asset)
    fig.savefig(ROOT/(asset+'.pdf'),bbox_inches='tight');fig.savefig(ROOT/(asset+'.png'),bbox_inches='tight');plt.close(fig)


def tex_escape(value):
    mapping={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}',
        '–':'--','≤':r'$\leq$','≥':r'$\geq$'}
    return ''.join(mapping.get(char,char) for char in str(value))


def table():
    frame=read('data').fillna('Unavailable');spec=read('specification').iloc[0]
    headers=list(frame.columns);widths=np.asarray([float(x) for x in str(spec.column_widths).split('|')]);widths=widths/widths.sum()
    tex_frame=frame.copy();resolution_note=''
    if 'Resolution' in tex_frame:
        tex_frame['Resolution']=tex_frame['Resolution'].map(lambda value:format(float(value),'.6g') if str(value)!='Unavailable' else value)
        resolution_note=' Resolution display is rounded to six significant digits; exact values remain in the local CSVs.'
    escaped=tex_frame.map(lambda value:tex_escape('N/A' if str(value)=='Unavailable' else value))
    columns='@{}'+''.join(r'>{\raggedright\arraybackslash}p{\dimexpr '+format(w,'.6f')+'\\linewidth-'+format(2*(len(widths)-1)*w,'.6f')+'\\tabcolsep\\relax}' for w in widths)+'@{}'
    header=' & '.join(tex_escape(h) for h in headers)+r' \\'
    lines=[r'\begin{longtable}{'+columns+'}',r'\caption{'+tex_escape(spec.title)+r'}\label{tab:'+str(spec.asset_id).removeprefix('table')+r'}\\',r'\toprule',
        header,r'\midrule',r'\endfirsthead',r'\multicolumn{'+str(len(headers))+r'}{l}{Table \thetable{} (continued)}\\',r'\toprule',header,r'\midrule',r'\endhead']
    lines+=[' & '.join(row)+r' \\' for row in escaped.itertuples(index=False,name=None)]
    missing_note=' N/A denotes an unavailable measurement.' if frame.astype(str).eq('Unavailable').any().any() else ''
    lines +=[r'\bottomrule',r'\end{longtable}',r'{\footnotesize '+tex_escape(str(spec.caption_note)+missing_note+resolution_note)+r'\par}']
    (ROOT/'table.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with PdfPages(ROOT/'table.pdf') as pdf:
        for page,start in enumerate(range(0,len(frame),18),1):
            f=frame.iloc[start:start+18]
            cells=[[textwrap.fill(str(value),max(10,int(135*widths[i]))) for i,value in enumerate(row)] for row in f.itertuples(index=False,name=None)]
            wrapped_headers=[textwrap.fill(h,max(10,int(135*w))) for h,w in zip(headers,widths)]
            row_lines=np.asarray([max(value.count('\n')+1 for value in row) for row in [wrapped_headers]+cells])
            row_inches=.15*row_lines+.12
            wrapped_note=textwrap.fill(str(spec.caption_note),170)
            note_inches=.15*(wrapped_note.count('\n')+1)+.2
            table_inches=float(row_inches.sum());height=table_inches+note_inches+1.0
            fig=plt.figure(figsize=(14,height),layout=None);fig.set_constrained_layout(False)
            ax=fig.add_axes([.025,(note_inches+.15)/height,.95,table_inches/height]);ax.axis('off')
            tab=ax.table(cellText=cells,colLabels=wrapped_headers,
                colWidths=widths,cellLoc='left',bbox=[0,0,1,1]);tab.auto_set_font_size(False);tab.set_fontsize(8)
            for (ri,ci),cell in tab.get_celld().items():
                cell.set_height(row_inches[ri]/table_inches);cell.set_edgecolor('#d6dce1');cell.set_linewidth(.35)
                if ri==0:cell.set_facecolor('#dce8ef');cell.set_text_props(weight='bold')
                elif ri%2==0:cell.set_facecolor('#f5f7f8')
            fig.text(.025,1-.15/height,str(spec.title)+((' — page '+str(page)) if len(frame)>18 else ''),va='top',fontweight='bold',fontsize=11)
            fig.text(.025,(note_inches-.05)/height,wrapped_note,va='top',fontsize=8,linespacing=1.3)
            pdf.savefig(fig,bbox_inches='tight');fig.savefig(ROOT/('table.png' if page==1 else 'table_page_'+str(page)+'.png'),bbox_inches='tight');plt.close(fig)


if __name__=='__main__':
    spec=read('specification').iloc[0];asset=str(spec.asset_id);ASSET=asset
    if asset.startswith('table'):table()
    else:main_figure(asset)
    pd.DataFrame([dict(file=k,sha256=v[0],rows=v[1]) for k,v in sorted(READS.items())]).to_csv(ROOT/'read_csv_hashes.csv',index=False)
