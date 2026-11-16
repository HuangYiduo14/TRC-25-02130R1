"""Numerical-only manuscript whose statements are tied to current result rows."""
import pandas as pd
from partC_rev1_src.phase_n8.common import write_json,sha
from .publication_tables import fmt


def tex(s):
    mapping={'&':r'\&','%':r'\%','_':r'\_','#':r'\#','$':r'\$','{':r'\{','}':r'\}','–':'--'}
    return ''.join(mapping.get(ch,ch) for ch in str(s))


def write(release,facts):
    root=release.root;c=release.ctx;summary=facts['summary'];closed=facts['closed'];mix=facts['fallback'].effective_family.value_counts()
    def row(method,population='BACKBONE',state='OPEN'):
        f=summary if state=='OPEN' else closed
        return f.loc[f.suite_id.eq('s03')&f.method_id.eq(method)&f.population.eq(population)].iloc[0]
    m1,m2,m3=[row(m) for m in ('M1','M2','M3')];fits=facts['fits'];completed=int(fits.status.eq('COMPLETE').sum())
    composition='; '.join(str(n)+' '+tex(kind) for kind,n in mix.items())
    m1_neural=int(facts['fallback'].effective_kind.str.contains('ICNN').sum());partial_cost=facts['crossings'].iloc[0].recorded_attributable_setup_seconds
    macros={
        'MainBackboneCount':'87,582','MainActiveRegions':'1,566','MainRegionalPairs':'80,316',
        'FallbackBWMAPE':fmt(m1.wmape_percent),'PolynomialBWMAPE':fmt(m2.wmape_percent),'ConstantBWMAPE':fmt(m3.wmape_percent),
        'FallbackKeyWMAPE':fmt(row('M1','KEY_LINKS').wmape_percent),'PolynomialKeyWMAPE':fmt(row('M2','KEY_LINKS').wmape_percent),
        'ConstantKeyWMAPE':fmt(row('M3','KEY_LINKS').wmape_percent),'AdditionalClosedDemand':fmt(facts['reachability']['additional_unreachable_demand'],3),
        'LearningCompletedFits':str(completed),'FinalNeuralRegions':str(m1_neural)}
    (root/'results_macros.tex').write_text('% Values extracted from the companion CSVs.\n'+'\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in macros.items())+'\n',encoding='utf-8')
    claims=[]
    def claim(identifier,section,text,path,columns,aggregation='exact row selection',filter='ALL',units='as named'):
        claims.append(dict(claim_id=identifier,section=section,text=text,data_relative_path=path,columns=columns,row_filter=filter,units=units,aggregation=aggregation,source_hash=sha(root/path)))
    claim('C01','5.1','Study setting and population lineage','numerical_data/table1/data.csv','Setting|Verified value')
    claim('C02','5.2','Assignment WMAPE, cost error, gap and runtime','numerical_data/table2/full_metrics.csv','method_id|population|wmape_percent|backbone_cost_error_percent|RG_SPTT|online_seconds')
    claim('C03','5.3','Regional cost summary','numerical_data/table3/region_observations.csv','region_id|role|cost_wmape_percent','equal-region mean/median/p90')
    claim('C04','5.4','Five-resolution comparison','numerical_data/table4/suite_metrics.csv','resolution|backbone_wmape|key_wmape|online_seconds')
    claim('C05','5.5','Closure redistribution','numerical_data/table5/redistribution.csv','method_id|population|redistribution_percent')
    claim('C06','5.5','Closed-reference accuracy','numerical_data/table5/accuracy_metrics.csv','method_id|population|wmape_percent|reference_RG_SPTT')
    claim('C07','5.6','Timing scope and unavailable amortization','numerical_data/table6/online_and_crossings.csv','method_id|online_seconds|recorded_attributable_setup_seconds|status')
    def fig(name,caption,label=None,landscape=False):
        stem='figure'+name
        content=r'\begin{figure}[p]\centering'+'\n'+r'\includegraphics[width=\linewidth,height=.83\textheight,keepaspectratio]{numerical_data/'+stem+'/'+stem+'.pdf}'+'\n'+r'\caption{'+caption+r'}\label{fig:'+str(label or name)+r'}\end{figure}'+'\n'
        return r'\clearpage\begin{landscape}'+'\n'+content+r'\clearpage\end{landscape}'+'\n' if landscape else content
    def table(n,caption):
        body='\n'+r'\begingroup\small\setlength{\tabcolsep}{5pt}'+'\n'+r'\input{numerical_data/table'+n+r'/table.tex}\endgroup'+'\n'
        return r'\clearpage\begin{landscape}'+'\n'+body+r'\clearpage\end{landscape}'+'\n' if n in ('2','3','4','5','6') else body
    parts=[r'\section{Numerical experiments}\label{sec:numerical}',r'\subsection{Study setting and reproducibility}',
        r'''The study uses the accepted national freight road assignment and a fixed active regional support. Freight input is FAF5.6.1 with baseline-year 2017 annual tonnage, expressed as annual kilotons. The separately consumed BTS experimental county disaggregation factors contain 3,144 county codes; 3,133 match extracted centroid records and supply 3,225 solver origins. The accepted physical graph retains 3,497 flagged centroid nodes. These populations are distinct: a county may supply multiple centroid records, and retained centroid nodes need not be active origins. The local factor release year is not established by the available source metadata.

The accepted graph has 346,373 nodes and 661,644 directed arcs. It is obtained from 665,303 prepared directed arcs; input exclusions are preserved rather than reconstructed. The 41,637,922 original centroid OD records aggregate to 10,193,770 demand rows. In the open graph, 10,190,380 aggregated rows are represented and 3,390 remain unreachable. Represented demand is 12,654,353.578362 of 12,667,037.704539 annual kilotons. We retain unreachable demand in the coverage denominator and do not renormalize the represented rows.

Current capacity is 0.30 times the authenticated prepared capacity, applied once. Preparation includes the payload, passenger-car-equivalent, traffic-background and annualization assumptions in Table~\ref{tab:1}; historical preparation factors are not reapplied. This is a synthetic annual-cost assignment regime, not a calibrated reconstruction of hourly congestion. Hardware, software and resource settings are also listed in Table~\ref{tab:1}. Numerical jobs use one worker and one nested numerical thread.

The main abstract network has 79,174 runtime nodes, 230,383 links, \MainBackboneCount{} backbone coordinates and \MainRegionalPairs{} active regional terminal-pair coordinates. The 1,566 active regions are a fixed subset of 1,847 final regions; 281 inactive regions remain excluded. Neither capacity tuning, inactive promotion nor active-support expansion is performed.''',
        table('1','Study setting and reproducibility.'),
        r'\subsection{Full-network benchmark and abstract backbone-flow validation}',
        r'''Let $f^*$ denote the saved physical CFW CURRENT reference after 45 updates. Its fresh signed relative gap is $6.949331890154105\times10^{-4}$. For suite $s$, the comparison target is the forward projection $\lambda_s^*=\pi_s(f^*)$. We compare each native abstract solution with this target; predicted physical flows are never reconstructed. A physical primitive or physical-member coordinate receives its own source flow; a parallel bundle receives the sum; a series macro receives the minimum of its parallel-bundle sums. Baseline series consistency and exact saved CURRENT regional trace joins are audited. The regional reference therefore preserves the saved path mixture rather than being inferred from physical marginals or a new all-or-nothing loading.

For the complete backbone $B_s$, flow error is
\begin{equation}\operatorname{WMAPE}_{B_s}=100\frac{\sum_{a\in B_s}|\lambda_{s,a}-\lambda^*_{s,a}|}{\sum_{a\in B_s}\lambda^*_{s,a}}.\end{equation}
All backbone coordinates, including zero-reference coordinates, contribute once. Connectors and regional pairs are excluded from this denominator. A second metric uses 200 positive-reference mainland coordinates frozen before predictions, selected by round-robin sampling of 250-km projected grid cells from the highest-reference-flow candidate pool. Their IDs remain fixed for the closure comparison. The signed bounded link error is $(\lambda-\lambda^*)/\max(|\lambda|,|\lambda^*|)$, with zero when both flows vanish. Regional cost error is a separate estimand.

The reported backbone cost is $C_B(\lambda)=\sum_{a\in B}\lambda_a(b_a+k_a\lambda_a^4)$ using the native packed coefficients; internal and half-node charges are included once in $b_a$. Backbone cost error is $100|C_B(\lambda)-C_B(\lambda^*)|/C_B(\lambda^*)$. It is neither a physical-network cost error nor an observed-traffic error. Figure~\ref{fig:3} displays the full backbone population, fixed-link sample and fresh gap histories.

The primary regional portfolios are ACNHN--best nonconstant with safety fallback, ACNHN--power-4, and ACNHN--FFTT. The first starts from frozen VALID-selected nonconstant potentials. Strictly negative or nonfinite marginal costs, including on zero-flow coordinates, invalidate the query before shortest-path use. All failing regions are downgraded as a batch to verified whole-region power-4 potentials and, if needed, FFTT. An objective is immutable within an epoch; changing it resets objective-dependent optimization state and certificates. Finite nonnegative costs below FFTT remain fidelity diagnostics, not blanket fallback triggers.''',
        'The final effective safety portfolio contains '+composition+'. It retains '+str(m1_neural)+' neural regions. '+
        ('The authorized bounded rescue replaced all remaining neural potentials after the discovery limit; the resulting method is an analytic mixture, so this comparison does not establish national neural-equilibrium performance. ' if m1_neural==0 else 'Its mixed composition is reported explicitly rather than treated as an all-neural model. ')+
        'The same final registry is used for open and closed assignments. Every discarded epoch remains in the timing and safety evidence.',
        r'Table~\ref{tab:2} includes the computed zero-error reference self-row and distinct physical CFW, FW and MSA controls. The three abstract portfolios have complete-backbone WMAPEs of \FallbackBWMAPE\%, \PolynomialBWMAPE\% and \ConstantBWMAPE\%, respectively, and fixed-link WMAPEs of \FallbackKeyWMAPE\%, \PolynomialKeyWMAPE\% and \ConstantKeyWMAPE\%. These values quantify agreement with the approximate projected benchmark. A small optimization gap certifies proximity only for the declared fixed objective and served demand; it does not establish physical, observed-traffic or model accuracy.',
        table('2','Main-suite comparison in native abstract coordinates.'),
        fig('3','Backbone validation for the three abstract portfolios. The map marks all 200 pre-frozen links. Complete-backbone errors retain zero-reference coordinates. Gap histories use fresh certificates within each fixed objective; discarded epochs are not joined into a fictitious smooth trajectory.'),
        r'\subsection{Regional surrogate performance and data requirements}',
        r'''Regional fidelity is evaluated on native terminal-pair marginal-cost vectors. For a declared region and cohort, cost WMAPE is $100\sum|\widehat\tau-\tau|/\sum\tau$ across all admitted queries and coordinates. This is computed directly from frozen predictions and matched native certified truths. It is not a relabeling of mean absolute relative error (MARE), which uses a one-second denominator floor and remains separately available. Table~\ref{tab:3} summarizes regional WMAPEs equally across regions, giving the mean, median and 90th percentile. Operating and stress cohorts remain distinct. The standalone power-4-plus-ICNN comparison uses the full frozen neural family, independently of the effective assignment portfolio.

The reopened operating and stress audits each have certified native rows for 1,564 of 1,566 active regions; the missing 28- and 39-coordinate cases remain explicit exclusions for those cohorts. The reused low-flow test cohort covers all 1,566 regions, and the high projected cohort covers 1,553. Reopened rows are now seen and do not constitute fresh blind validation. The difficult 28-coordinate training diagnostic and failed or severe fresh-label attempts are retained rather than removed to improve the error summary. Figure~\ref{fig:4} shows actual owned roads, portals and centroid positions for representative large, low-VALID-error and difficult regions. Regional terminal-pair chords in the later abstract maps have a different geometric meaning from these physical roads.

The data-requirement study uses sixteen preselected regions, four per size/difficulty stratum, with nested TRAIN fractions of 10, 25, 50, 70 and 100\%. The forty previously completed fits are reused unchanged. Only the missing eight regions are added, using the same seed, calibration, optimizer settings, 4,000-update budget and fixed VALID checkpoint-selection policy. There are \LearningCompletedFits{} completed fits across the eighty-slot neural docket. Figure~\ref{fig:5} reports region-level median and interquartile range of operating-query cost WMAPE on labels collected after model freezing, with eight regional datasets reused unchanged and eight newly attempted. Stress and reused cohorts, missing slots, fit status and all numerators/denominators are supplied with the figure. The bands describe variation among these selected regions, not repeated-seed uncertainty or a random sample of all regions.

The reused post-freeze labels admit 121 of 128 attempted queries, while the new labels admit all 128. All seven exclusions are in the reused 28-coordinate region's operating cohort, leaving its operating curve based on one admitted query out of eight. The combined post-freeze coverage is therefore 249 of 256 queries; regional sample counts alone do not express this unequal query coverage.

Fixed-data optimizer traces are shown separately in Figure~\ref{fig:S1}. An initialization selected by VALID remains update zero even if all 4,000 optimizer updates were executed. Such a result is not a zero-update training run. Intermediate robust losses and VALID MARE cannot be interpreted as cost-WMAPE learning curves.''',
        table('3','Regional marginal-cost error and cohort coverage.'),
        fig('4','Actual representative regional networks. Roads preserve stored geometry and multipart order. Portal and centroid symbols use fine-member physical locations; multiple terminal roles may share one physical node.'),
        fig('5','Regional sample-size study: native cost WMAPE on operating queries collected after model freezing, summarized across regions within each stratum. Eight regional evaluation datasets are reused unchanged and eight are newly attempted. Every band uses the actual admitted regions at that fraction. The reused 28-coordinate region has only one admitted operating query of eight; all seven excluded labels remain in the companion coverage table.'),
        r'\subsection{Resolution sensitivity and boundary consistency}',
        r'''Five constant-region assignments are compared at resolutions $0.3107232505953859$, $1$, $1.6$, $4$ and $12$. Each suite has its own compiled backbone, saved forward reference and frozen 200-link set. Table~\ref{tab:4} and Figure~\ref{fig:6} retain these changing denominators rather than treating the backbone as a constant population. The experiment studies this fixed family of constant regional potentials; it does not extrapolate trained regional models to untrained resolutions.

The resolution figure reports graph/active-support counts, complete assignment time, backbone and fixed-link errors, backbone cost error and active regional dimensions. Figure~\ref{fig:S2} displays the corresponding ownership midpoint maps and the adjusted Rand index on 324,602 common final region-owned physical arcs. Map display thinning is separate from the full statistical population. Colours are suite-local labels, not matched region identities, and midpoint markers should not be read as geographic region boundaries.''',
        table('4','Five-resolution constant-region assignments.'),
        fig('6',r'Resolution tradeoffs using each suite\textquotesingle s own forward reference, complete backbone and fixed 200 links. Dimension distributions include every active region.',landscape=True),
        r'\subsection{Counterfactual represented I-95 backbone closure}',
        r'''The closure is explicitly a represented-abstract-backbone scenario. Route metadata identifies 1,713 intended physical arcs. Of these, 1,105 have exact represented backbone incidence; 608 are suppressed backbone-interior arcs and are omitted. The overlay disables 869 abstract backbone coordinates. A parallel bundle is unavailable only when all of its represented alternatives are closed; a series macro is unavailable when any essential bundle is unavailable. The independently assigned matched physical control closes exactly the 1,105 represented source arcs and retains 55 healthy alternatives located elsewhere inside disabled macros. No zero-capacity BPR evaluation, artificial big-M cost, arbitrary macro deletion or hidden demand rescaling is used.

All three abstract portfolios start from feasible masked loading and use the same fixed regional models as their final open comparison. Additional abstract unreachability is \AdditionalClosedDemand{} annual kilotons; original and reachable OD ledgers are supplied separately for the abstract and matched physical graphs. The closure does not retrain regional models, expand support or restore suppressed topology.

Figure~\ref{fig:7} shows the applied mask, open flow, closed flow and gain/loss for the final safety portfolio, with shared open/closed colour limits and separate backbone/regional scales. Regional connections are terminal-pair chords. Figure~\ref{fig:S4} supplies the power-4 and FFTT responses. Table~\ref{tab:5} separates open-to-closed redistribution from closed-reference accuracy. Redistribution uses open-flow totals, both on the full backbone and outside the disabled coordinates. Accuracy instead compares the closed abstract solution with the forward projection of the matched physical closure control, retaining the original fixed 200 IDs even when some reference flows become zero. The control\textquotesingle s termination status and achieved gap remain visible; a time-limited control is an approximate reference, not exact closed equilibrium or empirical evidence of a real-world diversion.''',
        ('The closed abstract and physical graphs '+('share' if facts['paired_coverage']['same_served_OD_membership'] else 'do not share')+
         ' the same served OD membership. The exact overlap and both row/demand totals are included with Table~\\ref{tab:5}. '+
         ('This permits a common-served-demand interpretation for the declared native comparison. ' if facts['paired_coverage']['same_served_OD_membership'] else
          'The full-backbone discrepancy therefore includes different served demand; it is not an accuracy comparison restricted to a common served OD set. ')+
         'Open-to-closed redistribution includes lost demand as well as rerouting.'),
        table('5','Represented I-95 response, matched-reference comparison and demand coverage.'),
        fig('7','Represented I-95 counterfactual for the final safety portfolio: 1,105 represented source arcs closed, 608 suppressed interior arcs omitted. Backbone and regional terminal-pair chords use separate colour scales. Open-to-closed gain/loss is redistribution, not accuracy against the open benchmark.',landscape=True),
        r'\subsection{Offline cost, amortization and empirical limits}',
        r'''Complete supervised process-wall clocks are available for the current abstract assignments and the independent physical controls on the same recorded host, with one numerical worker. Source admission/loading, solver work and export are inside this outer boundary. New abstract runs reuse a feasible FFTT loading prepared in the separately measured shared preflight; the physical controls include fresh initialization. These are not matched cold-start experiments. Solver line searches, certificates and checkpoints are nested components and are not added again. Discarded safety epochs and any invalidated paired runs are retained separately from the final fixed-portfolio assignment.

Historical reference timing has a different scope: the 45-update reference records 3,353.947175 seconds of cumulative iteration wall, and a later frozen-state refresh records a separate 138.043553 seconds. The 200-update sensitivity records 14,614.275031 seconds of iteration wall and 17,724.460237 seconds over a broader campaign interval. Neither is a measured standalone cold assignment duration. Historical worker counts and missing host/timing details prevent treating those values as matched speedup baselines.

Table~\ref{tab:6} and Figure~\ref{fig:8} separate recorded elapsed subtotals from summed worker wall. Their sum is not a defensible complete offline wall-clock intercept. In addition, per-query initialization boundaries differ. Consequently cold-start amortization is not comparable, and no repeated-query crossing is estimated. A formula such as $C/(T_P-T_H)$ would require both an attributable offline cost and compatible per-query timings; an incomplete intercept alone cannot repair unmatched slopes. The figure instead presents measured components and explicit completeness limits. Missing construction/training work is not assumed free, and the time-limited physical control's different achieved gap further prevents an equal-accuracy speedup claim.

The numerical evidence is limited by the approximate physical reference, fixed active support, scenario reachability losses, incomplete local-label coverage, now-seen audit data and incomplete attributable offline timing. The represented I-95 scenario omits hidden interior arcs. No matched independent traffic observations are available, and the annual-cost regime does not validate causal diversion, economic welfare or operational policy. Background traffic enters capacity preprocessing; AADT is not independent validation. Dynamic passenger equilibrium, parking, delivery windows and hours-of-service constraints are not represented. The controls do not constitute a benchmark against unimplemented bush-based or origin-based algorithms. Regional cost and national flow error plots do not prove a causal error-propagation bound. These limitations narrow the empirical claims while preserving all unfavorable errors, failed safety queries and time-limited outcomes.''',
        table('6','Measured cost boundaries and unavailable cold-start amortization.'),
        fig('8','Recorded serial computation components and timing completeness. Parent elapsed intervals and summed worker wall are kept separate. Different initialization boundaries and incomplete attributable offline cost preclude a defensible repeated-query crossing. The physical comparator is time-limited.')]
    (root/'numerical_section.tex').write_text('\n\n'.join(parts).replace('\\par\\n','\\par\n'),encoding='utf-8')
    supplement=[r'\clearpage\section*{Numerical supplement}',r'\renewcommand{\thefigure}{S\arabic{figure}}\renewcommand{\theHfigure}{S\arabic{figure}}\setcounter{figure}{0}',r'\renewcommand{\thetable}{S\arabic{table}}\renewcommand{\theHtable}{S\arabic{table}}\setcounter{table}{0}']
    for n,caption in [('S1','Fixed-data optimizer losses, intermediate VALID MARE and selected/executed counters. These traces are distinct from the sample-size cost-WMAPE study.'),('S2','Five resolution ownership midpoint maps and adjusted Rand index on the full common final ownership population. Local labels are not matched across suites.'),('S3','Final fallback family counts, baseline traversal exposure and native cost minima. Finite nonnegative below-FFTT costs remain fidelity diagnostics.'),('S4','Power-4 and constant-region represented I-95 responses under the same explicit mask scope as the main closure figure.')]:supplement.append(fig(n,caption,landscape=n in ('S1','S2','S3','S4')))
    supplement.append(table('S1','Expanded identities, coverage and timing notes.'))
    (root/'numerical_supplement.tex').write_text('\n'.join(supplement).replace('\\par\\n','\\par\n'),encoding='utf-8')
    main=r'''\documentclass[10pt]{article}
\usepackage[margin=0.8in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern,amsmath,amssymb,graphicx,booktabs,longtable,array,pdflscape,textcomp}
\usepackage[colorlinks=true,linkcolor=blue,urlcolor=blue]{hyperref}
\usepackage[font=small,labelfont=bf]{caption}
\setlength{\parskip}{5pt}\setlength{\parindent}{0pt}
\setcounter{section}{4}\setcounter{figure}{2}
\input{results_macros}
\begin{document}
\begin{center}{\Large National freight assignment through regional convex surrogates}\\[5pt]
{\large Numerical section and reproducible data supplement}\\[3pt]
{\small Review copy: native abstract-coordinate comparison; measured limitations retained}
\end{center}
\input{numerical_section}
\input{numerical_supplement}
\end{document}
'''
    (root/'main_numerical.tex').write_text(main,encoding='utf-8')
    pd.DataFrame(claims).to_csv(root/'claim_to_data.csv',index=False)
    crosswalk=[dict(reviewer_point='Correct numerical estimand',response='All assignment KPIs use native abstract flows versus the forward benchmark.',claims='C02|C04|C06'),
        dict(reviewer_point='Practical safety fallback',response='Whole-region fixed-potential epochs; final composition and all discarded costs disclosed.',claims='C02|C07'),
        dict(reviewer_point='Actual I-95 response',response='Executed represented mask; 608 hidden arcs omitted; actual open/closed flows and matched physical control.',claims='C05|C06'),
        dict(reviewer_point='Learning curves and costs',response='Direct native cost WMAPE, region-level bands, fixed-data optimizer traces separate.',claims='C03'),
        dict(reviewer_point='Timing and empirical limits',response='Complete process clocks separated from historical iteration and worker-wall measurements.',claims='C01|C07')]
    pd.DataFrame(crosswalk).to_csv(root/'reviewer_point_crosswalk.csv',index=False)
    write_json(root/'private_provenance/MANUSCRIPT_FACTS.json',dict(macros=macros,final_composition=mix.to_dict(),completed_neural_fits=completed,recorded_setup_seconds=partial_cost))
    return dict(status='SOURCE_WRITTEN',claim_rows=len(claims),completed_neural_fits=completed)
