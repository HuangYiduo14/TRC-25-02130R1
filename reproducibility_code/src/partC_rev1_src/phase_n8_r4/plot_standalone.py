"""Independent CSV-only drawing layer. Run with this directory as its sole input."""
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
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator, ScalarFormatter, SymmetricalLogLocator, FuncFormatter

ROOT=Path(__file__).resolve().parent
COLORS=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9','#333333','#999933','#882255']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,
    'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,
    'savefig.facecolor':'white','axes.grid':False})
READS={}
SCENARIO_ORDER=['FFTT_s02','FFTT_s03','FFTT_s04','FFTT_s05','FFTT_s06','S03_ANALYTIC_VALID',
    'N8R4_PRIMARY_BOUND_SAFE_V1','N8R4_BEST_ANY_BOUND_SAFE_V1','N8R4_NONCONSTANT_SOURCE_BOUND_SAFE_V1',
    'PHYSICAL_CFW','PHYSICAL_FW','PHYSICAL_MSA']
ROLE_STYLE={'BACN':(COLORS[0],'o'),'DESTINATION_CENTROID':(COLORS[1],'s'),
    'ORIGIN_CENTROID':(COLORS[2],'^'),'RACN':(COLORS[3],'D')}


def read(name):
    if '|' in name:return pd.concat([read(part) for part in name.split('|')],ignore_index=True)
    p=(ROOT/name).resolve(strict=True)
    if p.parent!=ROOT or p.suffix.lower()!='.csv':raise ValueError('Only immediate local CSV drawing data may be read')
    READS[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    return pd.read_csv(p,dtype=str,keep_default_na=False)


def num(data,col):
    return pd.to_numeric(data[col],errors='coerce').to_numpy(float)


def groups(data,key):
    return list(data.groupby(key,sort=False,dropna=False)) if key else [('',data)]


def legend(ax):
    handles,labels=ax.get_legend_handles_labels()
    unique=dict(zip(labels,handles))
    if unique:ax.legend(unique.values(),[textwrap.fill(str(label).replace('CONVERGED_REPRESENTED_DEMAND_ONLY','converged (represented)').replace('TIME_LIMIT','time limit'),34) for label in unique],frameon=False,fontsize=7,
        loc='upper left',bbox_to_anchor=(1.01,1),borderaxespad=0)


def symlog_label(value,position):
    magnitude=abs(value)
    if magnitude and (magnitude>=1000 or magnitude<.001):
        exponent=np.log10(magnitude)
        if np.isclose(exponent,round(exponent)):
            return ('$-' if value<0 else '$')+'10^{'+str(int(round(exponent)))+'}$'
    return format(value,'.4g')


def axes_style(ax,spec):
    ax.set_title(spec['title'],loc='left',pad=12)
    ax.set_xlabel(spec.get('xlabel',''));ax.set_ylabel(spec.get('ylabel',''))
    for axis in ('x','y'):
        scale=spec.get(axis+'scale','linear') or 'linear'
        kwargs={'linthresh':float(spec[axis+'linthresh'])} if scale=='symlog' and spec.get(axis+'linthresh') else {}
        # Reapplying a linear scale resets categorical FixedLocator/Formatter.
        if scale!='linear':getattr(ax,'set_'+axis+'scale')(scale,**kwargs)
        if scale=='symlog':
            threshold=kwargs.get('linthresh',2.)
            lo,hi=getattr(ax.dataLim,'interval'+axis)
            target=getattr(ax,axis+'axis')
            if np.isfinite([lo,hi]).all():
                if lo>=0:
                    limits=getattr(ax,'get_'+axis+'lim')()
                    getattr(ax,'set_'+axis+'lim')(max(0.,limits[0]),limits[1])
                if max(abs(lo),abs(hi))<=threshold:
                    target.set_major_locator(MaxNLocator(nbins=5,min_n_ticks=3))
                    target.set_major_formatter(ScalarFormatter(useMathText=True))
                else:
                    decades=np.log10(max(abs(lo),abs(hi))/threshold)
                    dense_ticks=decades<=(2 if axis=='x' else 4)
                    target.set_major_locator(SymmetricalLogLocator(base=10,linthresh=threshold,subs=[1,2,5] if dense_ticks else [1]))
                    target.set_major_formatter(FuncFormatter(symlog_label))
                    left,right=getattr(ax,'get_'+axis+'lim')()
                    visible=[v for v in target.get_majorticklocs() if left<=v<=right]
                    if len(visible)<2:target.set_major_locator(MaxNLocator(nbins=5,min_n_ticks=3))
    if spec.get('note'):
        ax.text(0,-.25,'\n'.join(textwrap.wrap(spec['note'],int(spec.get('note_wrap_width') or 100))),transform=ax.transAxes,
            fontsize=float(spec.get('note_fontsize') or 7),va='top',ha='left',gid='panel_note')


def draw(spec):
    data=read(spec['data']);kind=spec['kind'];group=spec.get('group','')
    fig,ax=plt.subplots(figsize=(6.6,4.5));records=[]
    x=spec.get('x','');y=spec.get('y','')
    if kind=='status':
        ax.axis('off')
        content='\n\n'.join('\n'.join(textwrap.wrap(row['text'],75)) for row in data.to_dict('records'))
        ax.text(0,1,content,transform=ax.transAxes,va='top',fontsize=11,linespacing=1.5)
        ax.set_title(spec['title'],loc='left',pad=16)
        records.append(dict(panel=spec['panel'],series='explicit disposition',input_rows=len(data),
            finite_plotted_rows=0,excluded_nonfinite_rows=0,x_sum=None,y_sum=None))
    elif kind in ('line','ecdf','scatter','tail','budget','optimization'):
        color_key=spec.get('color_group','') or group
        keys=list(dict.fromkeys(data[color_key])) if color_key else ['']
        colors={k:COLORS[i%len(COLORS)] for i,k in enumerate(keys)}
        for i,(name,v) in enumerate(groups(data,group)):
            xx=num(v,x);yy=num(v,y) if y else np.empty(0)
            key=v[color_key].iloc[0] if color_key else '';style_index=keys.index(key)
            if 'scenario' in v and v.scenario.nunique()==1 and v.scenario.iloc[0] in SCENARIO_ORDER:
                style_index=SCENARIO_ORDER.index(v.scenario.iloc[0])
            color=COLORS[style_index%len(COLORS)];line_style=['-','--',':','-.'][(style_index//len(COLORS))%4]
            label=key if color_key else name
            bin_audit={}
            if kind=='ecdf':
                take=np.isfinite(xx);xx=np.sort(xx[take]);yy=np.arange(1,len(xx)+1)/max(1,len(xx))
            elif kind=='tail':
                take=np.isfinite(xx)&(xx>=0);values=np.sort(xx[take])[::-1]
                xx=np.arange(1,len(values)+1)/max(1,len(values))
                yy=np.cumsum(values)/values.sum() if values.sum()>0 else np.zeros(len(values))
            else:
                take=np.isfinite(xx)&np.isfinite(yy);xx=xx[take];yy=yy[take]
                order=np.argsort(xx,kind='stable');xx=xx[order];yy=yy[order]
            if kind=='optimization' and spec.get('bin_width'):
                width=float(spec['bin_width']);bucket=np.floor((xx-1)/width)
                temp=pd.DataFrame({'bucket':bucket,'x':xx,'y':yy}).groupby('bucket',sort=True).agg(
                    x=('x','max'),y=('y','mean'),count=('y','size'))
                xx=temp.x.to_numpy(float);yy=temp.y.to_numpy(float)
                bin_audit=dict(aggregation_rule='Updates 1..width, width+1..2width; x is last actual update in each bin',
                    minimum_bin_rows=int(temp['count'].min()),maximum_bin_rows=int(temp['count'].max()),last_bin_rows=int(temp['count'].iloc[-1]))
            if kind=='scatter':ax.scatter(xx,yy,s=12,alpha=.65,c=color,label=label,rasterized=True)
            else:
                alpha=.45 if kind in ('budget','optimization') else .9
                ax.plot(xx,yy,label=label,color=color,linestyle=line_style,alpha=alpha,lw=1.1,
                    marker='o' if kind=='budget' else None,markersize=3)
                if spec.get('mark_terminal')=='true':
                    for _,end in v.loc[v.status.ne('FRESH_INTERMEDIATE')].iterrows():
                        ex=float(end[x]);ey=float(end[y]) if end[y] else np.nan
                        if np.isfinite(ex) and np.isfinite(ey) and ey>0:
                            marker='x' if end.status=='TIME_LIMIT' else 'o'
                            ax.scatter([ex],[ey],c=color,marker=marker,s=32,zorder=4)
            records.append(dict(panel=spec['panel'],series=name,input_rows=len(v),finite_plotted_rows=len(xx),
                excluded_nonfinite_rows=len(v)-int(take.sum()),x_sum=float(xx.sum()),y_sum=float(yy.sum()),**bin_audit))
        axes_style(ax,spec);legend(ax)
    elif kind in ('bar','errorbar'):
        fig.set_size_inches(6.6,max(4.5,.24*len(data)+1.5))
        positions=np.arange(len(data));values=num(data,y);labels=data[x].tolist()
        bar_colors=[]
        for i,row in enumerate(data.to_dict('records')):
            scenario=row.get('scenario','');suite=row.get('suite','')
            style_index=SCENARIO_ORDER.index(scenario) if scenario in SCENARIO_ORDER else int(suite[1:])-2 if suite in ('s02','s03','s04','s05','s06') else i
            bar_colors.append(COLORS[style_index%len(COLORS)])
        if kind=='bar':ax.barh(positions,values,color=bar_colors,height=.7)
        else:
            low=num(data,spec['low']);high=num(data,spec['high'])
            for i in range(len(data)):
                ax.plot([low[i],high[i]],[positions[i],positions[i]],color=bar_colors[i],lw=1.4)
                ax.plot(values[i],positions[i],'o',color=bar_colors[i],ms=4)
        ax.set_yticks(positions,labels,fontsize=float(spec.get('bar_label_fontsize') or 7));ax.invert_yaxis();axes_style(ax,spec)
        records.append(dict(panel=spec['panel'],series='all',input_rows=len(data),finite_plotted_rows=int(np.isfinite(values).sum()),
            excluded_nonfinite_rows=int((~np.isfinite(values)).sum()),x_sum=None,y_sum=float(np.nansum(values))))
    elif kind in ('map','ownership'):
        if kind=='ownership':
            facets=groups(data,spec['facet']);plt.close(fig)
            columns=2;rows=(len(facets)+columns-1)//columns
            fig,axs=plt.subplots(rows,columns,figsize=(6.6,7.2),squeeze=False)
            for ax,(facet,v) in zip(axs.flat,facets):
                # Codes are categorical ownership identifiers, never numeric region quantities.
                backbone=v.is_final_backbone.str.lower().eq('true')
                ax.scatter(num(v.loc[backbone],x),num(v.loc[backbone],y),c='#aaaaaa',s=.6,linewidths=0,rasterized=True)
                regional=v.loc[~backbone];codes=pd.factorize(regional[spec['group']],sort=True)[0]
                ax.scatter(num(regional,x),num(regional,y),c=codes%20,cmap='tab20',s=1,linewidths=0,rasterized=True)
                ax.set_title(facet);ax.set_aspect('equal');ax.set_axis_off()
            for ax in list(axs.flat)[len(facets):]:ax.set_axis_off()
            fig.subplots_adjust(left=.02,right=.98,top=.91,bottom=.09,hspace=.23,wspace=.05)
            fig.suptitle(spec['title'],x=.02,y=.98,ha='left',fontsize=11)
            if spec.get('note'):fig.text(.02,.02,'\n'.join(textwrap.wrap(spec['note'],100)),fontsize=7)
        else:
            segments=[];values=[]
            for key,v in groups(data,group):
                v=v.sort_values('vertex_order',key=lambda s:pd.to_numeric(s,errors='raise'))
                xx=num(v,x);yy=num(v,y)
                if not (np.isfinite(xx).all() and np.isfinite(yy).all()):raise ValueError('Nonfinite map geometry')
                segments.append(np.column_stack([xx,yy]))
                if spec.get('value'):values.append(float(v[spec['value']].iloc[0]))
            lc=LineCollection(segments,linewidths=1.15,color='#777777' if not values else None)
            if values:
                a=np.asarray(values);lc.set_array(a);lc.set_cmap('viridis');lc.set_norm(Normalize(a.min(),a.max()))
                fig.colorbar(lc,ax=ax,pad=.02,shrink=.72,label=spec.get('value_label',spec['value']))
            ax.add_collection(lc);ax.autoscale();ax.set_aspect('equal')
            if spec.get('vertex_markers')=='true':
                locations=np.array([segment[len(segment)//2] for segment in segments])
                ax.scatter(locations[:,0],locations[:,1],s=12,c='#555555',linewidths=0,zorder=2,
                    label='One supplied road vertex per sampled segment')
                legend(ax)
            overlay=spec.get('overlay','')
            if overlay:
                points=read(overlay)
                for i,(name,v) in enumerate(groups(points,spec.get('point_group',''))):
                    color,marker=ROLE_STYLE.get(name,(COLORS[i%len(COLORS)],['o','s','^','D','v','P'][i%6]))
                    ax.scatter(num(v,'x'),num(v,'y'),label=name,c=color,
                        marker=marker,s=19,edgecolors='white',linewidths=.25,zorder=3)
                legend(ax)
            axes_style(ax,spec)
            ax.ticklabel_format(axis='both',style='plain',useOffset=False)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3));ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            for label in ax.get_xticklabels():label.set_rotation(60);label.set_ha('right');label.set_fontsize(8)
        records.append(dict(panel=spec['panel'],series='geographic observations',input_rows=len(data),finite_plotted_rows=len(data),
            excluded_nonfinite_rows=0,x_sum=float(np.sum(num(data,x))),y_sum=float(np.sum(num(data,y)))))
    elif kind=='heatmap':
        xx=list(dict.fromkeys(data[x]));yy=list(dict.fromkeys(data[y]));matrix=np.full((len(yy),len(xx)),np.nan)
        for row in data.to_dict('records'):
            matrix[yy.index(row[y]),xx.index(row[x])]=float(row[spec['value']])
        artist=ax.imshow(matrix,vmin=0,vmax=1,cmap='Blues',aspect='equal')
        for iy in range(len(yy)):
            for ix in range(len(xx)):
                if np.isfinite(matrix[iy,ix]):ax.text(ix,iy,f'{matrix[iy,ix]:.3f}',ha='center',va='center',fontsize=8,
                    color='white' if matrix[iy,ix]>.6 else 'black')
        ax.set_xticks(range(len(xx)),xx);ax.set_yticks(range(len(yy)),yy)
        fig.colorbar(artist,ax=ax,label=spec.get('value_label',''));axes_style(ax,spec)
        records.append(dict(panel=spec['panel'],series='stable-ID pairwise matrix',input_rows=len(data),
            finite_plotted_rows=int(np.isfinite(matrix).sum()),excluded_nonfinite_rows=int((~np.isfinite(matrix)).sum()),
            x_sum=None,y_sum=float(np.nansum(matrix))))
    else:raise ValueError('Unknown drawing kind: '+kind)
    if kind!='ownership':fig.tight_layout()
    if (kind=='map' or spec.get('note_after_axis')=='true') and spec.get('note'):
        # Equal-aspect maps can have narrow axes. Place the note after layout,
        # below the measured rotated ticks and xlabel rather than at a fixed y.
        fig.canvas.draw();renderer=fig.canvas.get_renderer()
        label_boxes=[label.get_window_extent(renderer) for label in ax.get_xticklabels() if label.get_visible()]
        label_boxes.append(ax.xaxis.label.get_window_extent(renderer))
        bottom=min(box.y0 for box in label_boxes)-12*fig.dpi/72
        note=next(text for text in ax.texts if text.get_gid()=='panel_note')
        note.set_y(ax.transAxes.inverted().transform((ax.bbox.x0,bottom))[1])
    for extension in ('pdf','png'):
        fig.savefig(ROOT/(spec['panel']+'.'+extension),dpi=200,bbox_inches='tight',metadata={'Creator':'Independent local CSV plot'} if extension=='pdf' else None)
    plt.close(fig)
    return records


def main():
    specs=read('plots.csv');records=[]
    for spec in specs.to_dict('records'):records.extend(draw(spec))
    pd.DataFrame(records).to_csv(ROOT/'rendered_series_audit.csv',index=False,float_format='%.17g')
    pd.DataFrame([dict(file=k,sha256=v) for k,v in READS.items()]).to_csv(ROOT/'read_csv_hashes.csv',index=False)
    print('CSV-only figure regeneration complete:',len(specs),'panels')


if __name__=='__main__':main()
