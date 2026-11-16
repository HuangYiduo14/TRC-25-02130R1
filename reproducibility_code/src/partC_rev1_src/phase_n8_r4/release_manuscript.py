"""Outcome-derived numerical text; every empirical number comes from a fact CSV."""
from pathlib import Path
import re
import pandas as pd
from .table_standalone import escape
from .release_tables import number

DIGITS=dict(zip('0123456789',['Zero','One','Two','Three','Four','Five','Six','Seven','Eight','Nine']))


def macro_name(key):return 'Rfour'+''.join(c if c.isalpha() else DIGITS.get(c,'') for c in key)
def macro(key):return '\\'+macro_name(key)+'{}'


def write(release,facts):
    root=release.root;lookup=facts.set_index('fact_id');commands=[];names=set()
    for row in facts.itertuples(index=False):
        name=macro_name(row.fact_id)
        if name in names:raise ValueError('Duplicate generated TeX fact macro')
        names.add(name)
        display=str(int(float(row.value))) if row.unit=='year' else number(row.value,digits=6)
        commands.append(chr(92)+'newcommand{'+chr(92)+name+'}{'+escape(display)+'}')
    (root/'results_macros.tex').write_text('% Generated only from numerical_facts.csv; never copied from old manuscript numbers.\n'+'\n'.join(commands)+'\n',encoding='utf-8')
    captions={
        'fig_3':'Physical-model comparison and fresh optimization histories. The physical comparison population is fixed before inspecting prediction errors. Distribution curves weight directed arcs equally; WMAPE uses reference-flow sums. Valid time-limited snapshots remain explicitly labeled. Historical and contemporary timing scopes are retained in the CSVs; no unconditional speedup follows from their juxtaposition.',
        'fig_4':'Deterministically selected representative subnetworks and diagnostic cases. Roads and fine-member portals/centroids use actual projected source coordinates. Region location markers are illustrative means of road vertices, not administrative centroids. Selection uses frozen TRAIN/VALID and active-dimension criteria, independently of the plotted final tests.',
        'fig_5':'New nested data-budget study. Each curve connects available distinct TRAIN counts for one region, seed, model and evaluation regime. Replacement sampling can expose fewer distinct inputs; actual exposure counts are reported separately. The optimizer policy, common calibration and initialization are fixed within the declared experiment. Curves need not be monotone. Missing labels and fits remain in the coverage inventories; empirical between-region/seed spread is not a confidence interval.',
        'fig_6':'Resolution sensitivity under constant-regional FFTT policies on the accepted suites. Backbone ownership changes with resolution. Accuracy uses the common recoverable physical-arc population, and stability uses common final region-owned arc IDs. These outcomes do not establish learned-model accuracy across resolutions or an optimal partition.',
        'fig_7':'Hypothetical I-95 closure audit. The intended physical geometry and unchanged open physical reference are measured. The fixed contracted representation cannot faithfully mask every intended movement; closed-flow and signed-change panels explicitly report the failed experiment. No arbitrary logical cut, baseline capacity change or inferred observed-event effect is substituted.',
        'fig_8':'Offline accounting and illustrative online-only repetition. Measured stage elapsed and summed disjoint worker wall have separate panels and are not added together. Complete offline intercepts, CPU/GPU hours and full break-even remain unestablished. Repetition slopes use qualified completed assignments and retain their individual timing scopes.',
        'fig_S1':'Fixed-data optimization diagnostics for the full TRAIN fraction. Raw batch losses, VALID checkpoints and replacement exposures are separate quantities. Only actually executed updates are shown; a zero-update or missing fit contributes a disposition rather than a fabricated curve.',
        'fig_S2':'Partition and interface diagnostics. Map markers are physical-arc endpoint midpoints rather than road boundaries. Categorical colors are local to each suite. Stability uses common stable IDs. Selected global-path stitching, local witness validity, full physical-centroid transit and repeated regional occurrences are audited separately; selected distinct-OD outcomes are not probability-sample lift rates.',
        'fig_S3':'Current regional error and safety observations. Source and bound-safe registries use the same certified queries within each cohort. Distributions of per-region query means, regional maxima and coordinate-frequency summaries have distinct denominators. E2 missing coverage remains separate from protected old-test coverage. The complete saved negative query illustrates a coupled derivative; it is not a population error bound.',
        'fig_S4':'Historical and new probability-sample path discrepancies. Historical demand and expansion weights remain separate. PPS draw units retain duplicates and receive no second demand weighting; bootstrap intervals describe this sampling design. The full error tails are retained. Frozen regional physical-cost paths are diagnostic shortest paths, not joint regional equilibrium labels.'}
    figure_files=[]
    for name,specs in release.specs.items():
        if not name.startswith('fig_'):continue
        lines=[]
        for i,spec in enumerate(specs):
            lines.extend([r'\begin{figure}[p]',r'\centering'])
            if i:lines.append(r'\ContinuedFloat')
            lines.append(r'\includegraphics[width=0.98\linewidth,height=0.73\textheight,keepaspectratio]{'+name+'/'+spec['panel']+'.pdf}')
            caption=escape(spec['panel'].replace('panel_','Panel ')+'. '+spec['title']+'. '+captions[name])
            lines.append(r'\caption{'+caption+'}')
            if not i:lines.append(r'\label{'+name+'}')
            lines.extend([r'\end{figure}',r'\clearpage'])
        p=root/(name+'_include.tex');p.write_text('\n'.join(lines)+'\n',encoding='utf-8');figure_files.append(p)
    m=macro
    section=r'''\section{Current-data numerical experiments}
\label{sec:current-numerical}
The experiments below evaluate the current frozen implementation and source data. They retain unfavorable outcomes, distinguish numerical approximation from observed-traffic validation, and preserve the accepted physical reference. The original neural portfolios, unchanged-model warm-start diagnostics and newly defined fixed bound-safe portfolios are distinct scenarios. The experiment-only learning models never replace the national portfolios.

\subsection{Study setting, provenance and reproducibility}
The consumed freight input is \texttt{@@FAFFile@@}, using base year @@BaseYear@@ and the \texttt{@@DemandColumn@@} field in annual kilotons. County disaggregation uses separate origin and destination factors; the available local evidence does not establish their release year. A factor-table row, county, extracted centroid and solver origin are different units. The preprocessing stage contains @@RawPhysicalLinks@@ physical links and @@Centroids@@ extracted centroids. The accepted assignment graph contains @@PhysicalNodes@@ nodes and @@PhysicalArcs@@ directed arcs. Its @@OriginalODRows@@ original demand records aggregate to @@AggregatedODRows@@ OD rows with @@Origins@@ physical origins. Of these, @@RepresentedODRows@@ aggregated ODs and @@RepresentedDemand@@ annual kilotons are represented; @@UnreachableODRows@@ ODs carrying @@UnreachableDemand@@ annual kilotons remain inherited unreachable demand. These exclusions are not counted as solved traffic.

Graph counts retain their construction stage. For s03, the original representation contains @@PrimaryBnodes@@ B nodes and @@PrimaryRnodes@@ R nodes, including @@PrimaryRcontrols@@ singleton controls. Its @@PrimaryTerminals@@ region-specific terminal copies include @@PrimaryOriginCopies@@ origin and @@PrimaryDestinationCopies@@ destination copies. The conceptual regional support has @@PrimaryConceptualPairs@@ pairs; the accepted active support used here has @@PrimaryActiveCoordinates@@ coordinates. After compilation and directional OD attachments, the assignment graph contains @@PrimaryRuntimeNodes@@ nodes and @@PrimaryRuntimeLinks@@ links. The @@PrimaryWrapperNodes@@ added OD wrapper nodes are distinct from regional centroid copies. Companion counts for every resolution accompany the setting table.

The current cost calculation multiplies the authenticated, already prepared capacity by @@CapacityMultiplier@@ once. Preparation subtracts nontruck and PCE-weighted truck background using the retained AADT matching and capacity rules. Its assumptions include payload @@Payload@@ tons per truck, truck PCE @@PCE@@, directional fraction @@DirectionSplit@@, K factor @@Kfactor@@, D factor @@Dfactor@@ and fallback peak-hour factor @@PHF@@. Missing truck AADT defaults to a fraction @@TruckAADTDefault@@ of total AADT, and the remaining-capacity floor uses fraction @@CapacityFloor@@. The matching summary contains @@AADTMatched@@ geometric matches and @@AADTDefaults@@ default AADT values. Defaults and close competitors prevent interpreting complete output coverage as complete measurement coverage. Capacity-calibration AADT is not independent validation.

Contemporary timed scientific jobs run serially with @@NumericWorkers@@ numerical worker and @@NestedThreads@@ nested thread on the recorded Windows environment. No GPU is used. ``Cold'' denotes a fresh feasible assignment initialization, not an empty operating-system or compilation cache. Complete supervised-job wall time starts before code snapshotting and child launch and includes input loading, execution and exit work; inherited assignment records and historical cumulative recorded iteration-wall timers retain their narrower scopes. The latter include checkpoint/refresh work inside recorded iterations and are not pure update/kernel clocks. Table~\ref{tab:settings} and the archive retain version, hardware and exact source identities.
\input{table_1/table.tex}

\subsection{Full-network benchmarks and physical-model comparisons}
The primary numerical reference is the accepted physical x45 state; x200 is retained as a sensitivity reference. The contemporary controls use the same physical graph, OD population, BPR costs, one-worker backend and feasible free-flow initial loading for FW, safeguarded conjugate FW and MSA. The nominal target is RG-SPTT at most @@NominalGap@@. Complete-job caps are @@CapCFW@@ seconds for safeguarded CFW, @@CapFW@@ for FW and @@CapMSA@@ for MSA; each safe-variant cap is @@SafeRunCap@@ seconds. These are resource bounds, not observed convergence times. Stopping uses a fresh signed gap,
\[
  \mathrm{RG}_{\mathrm{SPTT}}=\frac{\langle f,t(f)\rangle-\mathrm{SPTT}(t(f))}{\mathrm{SPTT}(t(f))},
\]
computed from the same state and cost vector. A small optimization gap certifies the implemented objective on represented demand; it does not establish physical approximation accuracy or independent traffic validity. The comparison does not include every scalable assignment method and makes no superiority claim over unrun alternatives.

The original neural failures are retained. Their exact native records show rejection at the first objective evaluation after a feasible AON initialization using the scenario costs evaluated at zero flow, before any accepted update. For the original primary portfolio, reopening the complete saved @@NegativeQueryDimension@@-coordinate regional query reproduces cost @@NegativeOriginalCost@@ seconds at zero-based native index @@NegativeCoordinate@@, even though that coordinate's own flow is @@NegativeOwnFlow@@. The coupled gradient depends on the other coordinates. The offending region's maximum native scale ratio is @@NegativeRegionScaleRatio@@; the portfolio-wide maximum is @@NegativeGlobalScaleRatio@@, a different statistic. Convexity alone does not force every gradient component to be nonnegative.

The fixed bound-safe variants attenuate the complete neural residual using a coefficient frozen from a weight-only lower bound. The transformation defines a different scalar potential; value, gradient and Hessian-vector products remain consistent. It performs no dynamic clipping, query-dependent switching or silent source-weight mutation. At the saved offending query, the primary transformed cost is @@NegativeSafeCost@@ seconds. This safety result does not imply accuracy, nor that every transformed cost exceeds its free-flow value. Table~\ref{tab:scenarios} reports every rejection, warm diagnostic, converged run and valid time-limited snapshot separately. Time-limited states are not assigned extrapolated observed convergence times.
The nonconstant-source safe portfolio reached the declared target on represented demand after @@ScenarioN8R4NONCONSTANTSOURCEBOUNDSAFEV1acceptedupdates@@ accepted updates. Its fresh gap was @@ScenarioN8R4NONCONSTANTSOURCEBOUNDSAFEV1RGSPTT@@ and complete supervised-job time was @@ScenarioN8R4NONCONSTANTSOURCEBOUNDSAFEV1measuredseconds@@ seconds. The primary and best-any safe portfolios remained time-limited after @@ScenarioN8R4PRIMARYBOUNDSAFEV1acceptedupdates@@ and @@ScenarioN8R4BESTANYBOUNDSAFEV1acceptedupdates@@ updates, with fresh gaps @@ScenarioN8R4PRIMARYBOUNDSAFEV1RGSPTT@@ and @@ScenarioN8R4BESTANYBOUNDSAFEV1RGSPTT@@. These outcomes concern distinct objectives and do not rank their physical approximation accuracy.

All three contemporary physical controls remained time-limited. CFW, FW and MSA stopped after @@ScenarioPHYSICALCFWacceptedupdates@@, @@ScenarioPHYSICALFWacceptedupdates@@ and @@ScenarioPHYSICALMSAacceptedupdates@@ updates, with fresh RG-SPTT values @@ScenarioPHYSICALCFWRGSPTT@@, @@ScenarioPHYSICALFWRGSPTT@@ and @@ScenarioPHYSICALMSARGSPTT@@, respectively. These controls supply no observed physical runtime to the declared target.

All @@WarmRejectedPortfolios@@ unchanged portfolios also rejected the same certified analytic warm-start state before accepting an update. Each recorded minimum cost @@WarmNegativeCost@@ seconds at zero-based native coordinate @@WarmNegativeCoordinate@@ in the offending region, and none obtained a fresh neural optimization certificate. The rejection was therefore not confined to the tested cold initialization; behavior from every possible starting state remains unestablished. Their reported job times are failed diagnostic wall times. The inherited recorded analytic assignment time of @@WarmHistoricalSeedSeconds@@ seconds is a separate prerequisite and is not included in those job timers.
\input{table_2/table.tex}

Physical-flow comparisons use only exact recoveries: explicit physical members, singleton retained primitives and certified strict-series members with common flow. Suppressed backbone interiors, unknown parallel-member allocations and unrecovered regional interiors are not physical zeros. The intersection across the five suites contains @@CommonArcs@@ directed arcs, representing fraction @@CommonFlowShare@@ of x45 arc-flow and fraction @@CommonFlowCostShare@@ of x45 flow--cost. These are arc-population coverage measures, not served-OD shares. On this fixed population,
\[
 \mathrm{WMAPE}=\frac{\sum_a|\widehat f_a-f^{45}_a|}{\sum_a|f^{45}_a|},\qquad
 \delta_a=\frac{\widehat f_a-f^{45}_a}{\max(|\widehat f_a|,|f^{45}_a|)}
\]
with the both-zero case defined as zero. Population TSTT uses original physical BPR costs at recovered flows; fixed coarse charges are excluded. Successful recovery on this subset does not prove that the entire coarse solution has a feasible physical lift. Figure~\ref{fig_3} gives the reference-selected geometry, full-population distributions, timing histories and reference sensitivity.

The new represented-demand PPS design contains @@PPSDraws@@ draws, @@PPSUniqueODs@@ distinct aggregated OD rows and @@PPSOrigins@@ origins. Duplicate draws are retained. Each draw receives equal weight because demand already enters its sampling probability. WREL is the ratio of summed absolute path-time errors to summed reference times; WMEANREL averages absolute relative errors with a one-second denominator floor. The primary safe solved snapshot has terminal status \texttt{@@ScenarioN8R4PRIMARYBOUNDSAFEV1status@@} and fresh RG-SPTT @@ScenarioN8R4PRIMARYBOUNDSAFEV1RGSPTT@@. At that state, PPS WREL is @@PrimaryPPSWrel@@ percent, with paired-draw 95\% bootstrap interval [@@PrimaryPPSLow@@, @@PrimaryPPSHigh@@] percent, conditional on finite paired paths; WMEANREL is @@PrimaryPPSWmeanrel@@ percent over @@PrimaryPPSPairs@@ finite paired draws. Unreachable or missing draws remain separate coverage outcomes outside this paired-error denominator. These are probability-sample estimates for the named state and design, not all-OD error bounds or accuracy at a certified converged equilibrium. Historical cohorts remain now-seen diagnostics with their original demand and expansion weights. Figure~\ref{fig_S4} retains untrimmed tails, coverage and the selected path-component audits.
\input{fig_3_include.tex}

\subsection{Regional surrogate performance and data-budget learning}
The current audit distinguishes @@ActiveRegions@@ active regions from the @@E2SourceRegions@@ regions for which certified E2 source rows were reopened. The @@E2MissingRegions@@ missing E2 regions account for @@E2MissingCoordinates@@ active coordinates; old-test coverage does not fill that denominator. Source availability does not imply completed predictions. Paired comparisons retain only region/cohort groups completed by every frozen registry on identical certified rows; the archive also supplies all available predictions and registry-specific completion counts. E2 operating/stress data and protected old tests are reused, now-seen diagnostics. Table~\ref{tab:regional} separates their denominators and reports equal-region means of query MARE. Figure~\ref{fig_S3} provides regional tails, safety and attenuation, together with the named m28 TRAIN-e2 case and large zero-update docket. The preserved standalone NN TRAIN-e2 headline is @@HistoricalNNTraineHeadline@@ percent across equally weighted active regions; it is not the named region's own error. The m28 region has mean TRAIN-e2 MARE @@HistoricalMtwentyEightError@@ percent and contributes @@HistoricalMtwentyEightShare@@ percent of the regional-error sum. The leave-m28-out value @@HistoricalLeaveMtwentyEightOut@@ percent is reported only as sensitivity and does not replace the headline. These historical standalone-model quantities remain separate from the current portfolio comparisons. A seen training case is not an independent test result.
\input{table_3/table.tex}

The primary portfolio's equal-region E2 operating MARE is @@RegionalPrimaryOperatingMARE@@ percent, compared with @@RegionalPrimarySafeOperatingMARE@@ percent for its fixed bound-safe variant. Their stress values are @@RegionalPrimaryStressMARE@@ and @@RegionalPrimarySafeStressMARE@@ percent. The original primary has @@RegionalPrimaryStressNegative@@ negative coordinate observations in stress, while its safe variant has @@RegionalPrimarySafeStressNegative@@. The safety transformation therefore removes these negative costs while increasing average error on these cohorts. It also leaves the named m28 accuracy pathology: the reopened current primary and safe portfolios have TRAIN-e2 MARE @@CurrentMtwentyEightPrimaryMARE@@ and @@CurrentMtwentyEightPrimarySafeMARE@@ percent on @@CurrentMtwentyEightPrimaryQueries@@ queries. These seen-query comparisons do not establish fresh generalization, and nonnegative costs are neither a full free-flow floor nor an accuracy guarantee.

The new bounded learning study uses @@LearningRegions@@ distinct regions selected from frozen large, small, low-VALID-error and difficult TRAIN/VALID strata, with @@LearningSeeds@@ seed(s) per region. It uses the current @@LearningArchitecture@@ model, the constant free-flow model and a relative power4 fit. The nested TRAIN fractions are @@LearningFractions@@. The policy holds calibration-based normalization and the full NN initialization fixed within a region/seed; the common calibration subset belongs to every fraction. Analytic fitting uses only the corresponding fraction. The configured NN budget is @@LearningUpdates@@ optimizer updates per fit with the same robust loss and fixed VALID checkpoint selection. Available distinct TRAIN counts, actually exposed distinct inputs, replacement exposure, executed updates, selected checkpoints and compute are retained. This identifies a particular training policy, not a universal effect of data quantity independent of computation.

The study completed @@LearningCompleteFits@@ of @@LearningPlannedFits@@ planned fit cases. Fixed VALID selection retained the initialization in @@LearningInitializationSelectedFits@@ cases across @@LearningInitializationSelectedRegions@@ distinct region(s) even though training updates were executed; @@LearningNonfinalSelectedFits@@ cases selected a checkpoint preceding the final iterate. After all experiment models froze, @@FreshLabelAB@@ of @@FreshLabelPlanned@@ planned local evaluation inputs obtained usable certified A/B labels. The named m28 case contributes only @@FreshMtwentyEightOperatingAB@@ of @@FreshMtwentyEightOperatingPlanned@@ operating queries and @@FreshMtwentyEightStressAB@@ of @@FreshMtwentyEightStressPlanned@@ stress queries. Its operating curve therefore has a particularly small evaluation denominator. The fresh-label exclusions comprise @@FreshLabelC@@ quality-C and @@FreshLabelQuarantined@@ quarantined inputs. Excluded labels contribute missing coverage, not a favorable replacement or zero error. Figure~\ref{fig_4} shows true selected-region geometry; Figure~\ref{fig_5} connects only measured data budgets. Region/seed spread is empirical and no monotonic trend is imposed. Fixed-data optimization histories are separated in Figure~\ref{fig_S1}. These experimental weights are diagnostic only and do not change national assignments.
\input{fig_4_include.tex}
\input{fig_5_include.tex}

\subsection{Resolution, interfaces and lifting}
The accepted s02--s06 suites test resolution sensitivity under constant-regional FFTT policies. Final region counts, active dimensions and backbone ownership all change. Adjusted Rand index and arithmetic normalized mutual information use the same physical arcs that remain region-owned in every suite; physical portal Jaccard uses stable physical-node IDs. Raw pre-correction partition statistics are retained separately. Consequently, the observed tradeoff cannot be interpreted as trained accuracy at every resolution, a pure region-count intervention or a universal optimal partition.
\input{table_4/table.tex}

The interface audit has terminal status \texttt{@@BoundaryStatus@@}. It records @@PositiveLocalWitnesses@@ validated positive-flow local coarse-pair witnesses at frozen physical x45 costs, preserving their terminal-role transitions. Global-path checks completed for @@LiftCompletedODs@@ of @@LiftSelectedODs@@ selected distinct diagnostic ODs; @@LiftContinuityVerified@@ completed stitches pass the implemented continuity/cost check, including @@LiftVerifiedSelfODs@@ empty self-OD path(s). Uncompleted diagnostics remain missing evidence, and national conservation is reported separately. These unique-OD counts are not probability-sample lift rates. Independently shortest local witnesses can be incompatible at a shared fine boundary; failure to stitch those choices does not prove that no alternative fine-state lift exists. Full physical-centroid transit is audited separately from continuity, and repeated regional visits retain their ordered occurrences. The accepted physical reference permits centroid transit; it is not silently rerouted. Frozen-regional-cost shortest paths are diagnostic constructions, not joint regional equilibrium labels. Figure~\ref{fig_S2} supplies the ownership and interface evidence.
\input{fig_6_include.tex}

\subsection{Hypothetical I-95 closure}
The intended Virginia--Florida I-95 mainline corridor is matched to actual source road class, route, state and directed physical IDs in Virginia, North Carolina, South Carolina, Georgia and Florida. Source direction AB denotes digitization direction and is not treated as verified northbound/southbound metadata. The intended overlay contains @@ClosureArcs@@ physical arcs. Of these, @@ClosureHiddenArcs@@ lie inside @@ClosureHiddenBnodes@@ suppressed backbone nodes. Deleting mapped inter-backbone links alone leaves those hidden movements implicitly available. The mapping audit therefore returns \texttt{@@ClosureStatus@@}; it does not run an arbitrary logical cut or change regional capacities/support to force a result.

Figure~\ref{fig_7} reports the audited geometry and unchanged open physical reference, with explicit failed-experiment panels for closed flow and signed change. No paired faithful closure assignment, additional-unreachability estimate, diversion result, ton-mile effect or observed-disruption validation is claimed. A faithful intervention would require a separately authorized representation change that preserves the intended physical movement semantics.
\input{fig_7_include.tex}

\subsection{Offline cost, repeat queries and limitations}
Table~\ref{tab:offline} and Figure~\ref{fig_8} display historical preprocessing, partition/compilation, label-generation and training subtotals. Current safety and experiment work is recorded separately in the R4 job clock ledger; its figure-folder copy is a pre-release snapshot, and the final root ledger and handoff receipt include later work. Historical cumulative counters are counted once, inherited optimizer and validation-check time is subtracted where justified, and resumed records are deduplicated or excluded as ambiguous. Summed worker wall, elapsed critical path, allocated campaign windows and CPU/GPU telemetry are different quantities. Historical physical cumulative recorded iteration-wall time includes checkpoint/refresh work within each recorded iteration: @@HistoricalXFortyFiveIterationWallSeconds@@ seconds through x45 and @@HistoricalXTwoHundredIterationWallSeconds@@ seconds through x200. The broader x200 campaign record is @@HistoricalXTwoHundredBroaderCampaignSeconds@@ seconds. The later x45 frozen-state refresh took @@HistoricalXFortyFiveLaterRefreshSeconds@@ seconds and executed zero assignment updates. Historical production used @@HistoricalXFortyFiveWorkers@@ workers; its hardware identity and complete x45 cold elapsed time are not established by the targeted metadata. Missing original scope prevents a complete offline elapsed or CPU/GPU total.
\input{table_5/table.tex}

For identical repeated queries, a complete comparison would use
\[
 C_{\rm phys}(Q)=C_{\rm phys,off}+QT_{\rm phys},\qquad
 C_{\rm hyb}(Q)=C_{\rm hyb,off}+QT_{\rm hyb}.
\]
Only when $T_{\rm hyb}<T_{\rm phys}$ and the timing boundaries are comparable is the usual break-even expression meaningful:
\[
 Q^*=\left\lceil\frac{\max(0,C_{\rm hyb,off}-C_{\rm phys,off})}{T_{\rm phys}-T_{\rm hyb}}\right\rceil.
\]
The complete intercepts are not established here. The supplied online-only repetition curves are explicitly labeled arithmetic scenarios, with inherited and new timing scopes kept separate. They exclude safety-rejected and time-limited runs from completed-query slopes. They do not establish a confident full amortization benefit.
\input{fig_8_include.tex}

The evidence addresses numerical approximation against a physical model. Available local WIM and Illinois AADT candidates remain unqualified for this campaign because current-arc alignment, year, units, completeness, independence from capacity calibration and usage conditions are unresolved. No independent national observation, cross-year experiment or observed-event validation was performed. Static annual freight also omits staging, hours-of-service, delivery windows and dynamic passenger/freight interactions. Old emissions, GREET and Baltimore extrapolations are omitted from this focused revision. Method definitions, theoretical proof scope and unrun literature comparisons remain explicit later-edit items rather than being declared resolved by these numerical experiments.

\paragraph{Data and code availability.}
The delivered review archive contains plot-source CSVs, independent drawing code, numerical tables, source/claim identities and the exact scoped experiment code. It is not the complete national source database. Figure folders regenerate outside the project from their own CSVs. A public deposit and repository link remain a user-approved future action; no external publication is represented as completed. The archive is identified as the current numerical review artifact~\cite{currentarchive}.
'''
    if lookup.loc['PrimaryPPSStatus','value']!='MEASURED':
        start=section.index('At that state, PPS WREL')
        end=section.index('Historical cohorts remain now-seen diagnostics',start)
        section=section[:start]+('The bounded path-evaluation stage did not produce the primary safe solved-state PPS estimate. '
            'Its WREL, WMEANREL and bootstrap interval are unavailable; the completed scenario/track results and unvisited '
            'cases remain explicit in the supplied path coverage ledger. No accuracy estimate is inferred from its optimization gap. ')+section[end:]
    for key in re.findall(r'@@([^@]+)@@',section):
        if key not in lookup.index:raise ValueError('Unsupported manuscript fact: '+key)
        section=section.replace('@@'+key+'@@',m(key))
    (root/'numerical_section.tex').write_text(section,encoding='utf-8')
    supplement=r'''\clearpage
\section*{Numerical supplement}
The supplement retains data roles, denominators, failed cases and detailed diagnostics. Its standalone figure folders contain the actual observations used for drawing. A fresh surrogate optimization certificate, local regional label quality, physical-flow recovery and full physical-path admissibility remain separate claims.
\subsection*{Frozen learning policy and objective}
Training uses @@LearningOptimizer@@ at learning rate @@LearningRate@@ in @@LearningDtype@@, with gradient-norm clipping at @@LearningClip@@ and nonnegative projection of the power coefficient and residual gate. The gate has no upper clamp. Recurrent and head weights use a nonnegative softplus parameterization; signed input skips remain signed. Adam weight decay is @@LearningWeightDecay@@, and no explicit L1/L2 parameter penalty or dropout is added. Batches contain @@LearningBatchLarge@@ draws for active dimension at least @@LearningBatchThreshold@@ and @@LearningBatchSmall@@ otherwise. Sampling is with replacement. If both classes exist, operating and stress inputs receive probability masses @@LearningOperatingMass@@ and @@LearningStressMass@@; within each class, present family labels receive equal mass and rows are uniform within each family. Batch losses average draws without inverse-probability correction. The objective therefore reflects this declared mixture, not a uniform distribution over unique training inputs.

For target cost $\tau_{ij}$, predicted cost $\widehat t_{ij}$ and free-flow cost $c^0_j$, define $e_{ij}=\widehat t_{ij}-\tau_{ij}$, $d_{ij}=\max(|\tau_{ij}|,c^0_j,1)$, $r_{ij}=\max(|\tau_{ij}-c^0_j|,0.05d_{ij})$, and $w_j=0.5/m+0.5LOW_j/\sum_kLOW_k$. Let $H$ be Huber loss with threshold @@LearningHuberDelta@@. The per-query loss is
\begin{align*}
 \mathcal L_i={}&@@LearningTotalWeight@@\sum_jw_jH(e_{ij}/d_{ij})
 +@@LearningResidualWeight@@\sum_jw_jH(e_{ij}/r_{ij})\\
 &+@@LearningValueWeight@@H(\Delta_i/v_i)
 +@@LearningFloorWeight@@\sum_jw_j\left[\frac{(c^0_j-\widehat t_{ij})_+}{\max(c^0_j,1)}\right]^2,
\end{align*}
where $L'_i=L_i-\epsilon_i$, $U'_i=U_i+\epsilon_i$, $\Delta_i=(L'_i-\widehat W_i)_++(\widehat W_i-U'_i)_+$ and $v_i=\max(|U'_i|,|u_i^\top c^0|,1)$. The last term is a soft floor penalty, not a hard gradient guarantee. VALID checks occur at initialization and every @@LearningValidPeriod@@ updates; the selected checkpoint minimizes the fixed mean of $0.5(\mathrm{MARE}_{equal}+\mathrm{MARE}_{LOW})+0.1$ times relative potential-interval violation across VALID cohorts. This score differs from the training loss. Complete optimizer settings and the frozen docket are supplied in the table-3 CSVs.

\subsection*{Fixed safety-qualified potential}
For the supported linear-head two-layer network, write $x=u/s$ and $R(x)=\psi(x)-\psi(0)-\nabla\psi(0)^\top x$. The source potential and its separately named safe variant are
\[
 W(u)=u^\top c^0+\sum_j\eta_js_jx_j^5/5+aR(x),\qquad
 W_{safe}(u)=u^\top c^0+\sum_j\eta_js_jx_j^5/5+\rho aR(x).
\]
With signed input matrices $A_1,A_2$, nonnegative recurrent weights $W$, nonnegative head weights $v$, and $g^0=\nabla\psi(0)$, the frozen coordinate bounds are
\[
 \ell=v^\top\min\{A_2+W\min(A_1,0),0\},\quad
 b_j=\frac{a}{s_j}\max(g^0_j-\ell_j,0),\quad
 \rho=\min\left\{1,\min_{j:b_j>0}\frac{0.99c^0_j}{b_j}\right\}.
\]
The empty inner minimum gives $\rho=1$. For finite coefficients, positive scales, nonnegative $u,c^0,\eta,a,W,v$ and a linear head, the full coupled gradient obeys $t^{safe}_j(u)\ge c^0_j-\rho b_j\ge0.01c^0_j$. This is a fraction of the free-flow floor, not the full floor, and does not guarantee approximation accuracy. Conservative arithmetic cushions and downward rounding of an interior $\rho$ guard floating-point implementation; they are not an interval-arithmetic proof. The native tests check anchors, derivatives, Hessian-vector products, convex curvature and Torch parity. Per-region source types, coefficients, bounds and any unsupported-head fallback disposition are supplied in the safety registry. Value, gradient and Hessian-vector products all use the same frozen coefficient; there is no query-dependent clipping or switching.
\clearpage
\setcounter{figure}{0}
\renewcommand{\thefigure}{S\arabic{figure}}
\input{fig_S1_include.tex}
\input{fig_S2_include.tex}
\input{fig_S3_include.tex}
\input{fig_S4_include.tex}
'''
    for key in re.findall(r'@@([^@]+)@@',supplement):
        if key not in lookup.index:raise ValueError('Unsupported supplement fact: '+key)
        supplement=supplement.replace('@@'+key+'@@',m(key))
    (root/'numerical_supplement.tex').write_text(supplement,encoding='utf-8')
    wrapper=r'''\documentclass[11pt]{article}
\usepackage[margin=0.8in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern,amsmath,amssymb,graphicx,booktabs,longtable,array,caption,subcaption}
\usepackage[hidelinks]{hyperref}
\usepackage{microtype}
\setlength{\emergencystretch}{3em}
\setlength{\parskip}{0.35em}
\input{results_macros.tex}
\title{Current-data numerical experiments and review evidence}
\author{N8R4 numerical revision}
\date{Data-derived review build}
\begin{document}
\maketitle
\begin{abstract}
This numerical-only revision reports the current frozen freight-network experiments, including neural safety failures, separately defined bound-safe potentials, matched bounded physical controls, common-population physical errors, probability-sample path diagnostics and a new regional learning study. Coverage, timing boundaries and unsuccessful experiments remain explicit. The intended I-95 intervention is reported as unsupported where the fixed representation cannot faithfully encode it. Independent observed-traffic validation and complete offline amortization remain unestablished.
\end{abstract}
\setcounter{section}{4}
\setcounter{figure}{2}
\input{numerical_section.tex}
\bibliographystyle{plain}
\bibliography{references}
\input{numerical_supplement.tex}
\end{document}
'''
    (root/'main_numerical.tex').write_text(wrapper,encoding='utf-8')
    (root/'references.bib').write_text('@misc{currentarchive,\n  title={N8R4 Current-Data Numerical Review Archive},\n  year={2026},\n  note={Versioned local review artifact; source identities and claim-to-result crosswalk included. No public deposit claimed.}\n}\n',encoding='utf-8')
    (root/'INSERTION_README.md').write_text('# Inserting the numerical revision\n\nKeep the existing method manuscript unchanged until its separately listed inconsistencies are resolved. Insert numerical_section.tex in Section 5, results_macros.tex in the preamble, and the numbered table and figure folders beside it. The supplied wrapper numbers the section as 5 and the first figure as 3. Supplement figures use S numbering.\n\nPanels use separate PDF files and readable review pages; the final journal layout can combine them without changing plot inputs. CSVs are the authoritative drawing inputs. Numerical prose uses generated facts, not old manuscript numerals. Recompile main_numerical.tex using the recorded isolated portable toolchain; inspect all pages and the log.\n',encoding='utf-8')
    return dict(manuscript_facts=len(facts),main_sections=6,figures=len(figure_files),original_method_files_modified=False)
