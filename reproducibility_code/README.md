# FAF-to-P-ACNHN reproducibility package

This code release reconstructs the computational workflow of **Regional network compression with input-convex neural cost surrogates for large-scale freight assignment**, using the manuscript delivered on September 30, 2026. Start with public FAF network/demand files and the listed auxiliary inputs, then build cleaned GIS/SQLite databases and directed arrays, solve the physical assignment, rebuild backbone costs and regional partitions, generate regional labels, fit the surrogate models and assign the abstract networks.

The release contains Python source, explicit configurations and a compact **frozen topology recipe**. It contains no raw national inputs, old physical-flow arrays, regional training datasets or pretrained weights. The primary region mapping, B-node quotient, bundle/macro membership and represented-I95 ID list preserve the final topology decisions without replaying obsolete parameter searches. All numerical flows, backbone coefficients, charges, labels and models are recomputed. Every bundled mapping/data file is under 10,000,000 bytes; both the expanded distributed payload and its ZIP are under 100,000,000 bytes. Downloaded and generated runtime data are additional storage and are excluded from the release manifest.

The separate manuscript figure/table-data folders reproduce publication graphics and tables. Those renderers, PDFs and plotting extracts are intentionally outside this computational release. The optional last two stages here recompute the learning experiments underlying Figure 5, without rendering figures.

## 1. Install the environment and check the extracted code

Extract into a short, new directory, such as `C:\acnhn_reproduce` or `~/acnhn_reproduce`, and run commands from that directory. Python 3.12.3 and the exact observed scientific package versions are declared in `requirements.txt` and `metadata/observed_environment.json`. CPU execution is supported; CUDA is optional. No ArcGIS license or LaTeX installation is required.

Using conda:

```text
conda env create -f environment.yml
conda activate acnhn-reproduce
python scripts/verify_package.py
python scripts/run_pipeline.py --config configs/self_check.json --run-id smoke_v1 --all
```

Alternatively, create a Python 3.12 virtual environment and run `python -m pip install -r requirements.txt`. Activate that environment before the same check commands. GDAL support for `OpenFileGDB` and `GPKG` is required; the pinned Pyogrio/Fiona wheels supply the relevant libraries. The self-check exercises small nonempty GIS, graph, OD, label and training examples. It writes its evidence under `runs/smoke_v1/` and does not read FAF downloads or historical project outputs. Source imports are resolved from this extracted package.

The numerical adapters use the unchanged project implementations under `src/partC_rev1_src/`. Use the documented adapters under `scripts/`; the source tree also retains shared historical modules whose campaign CLIs depend on archival control documents. Those campaign CLIs are not this release's entrypoints.

## 2. Download and place the raw inputs

Official sources were checked on September 30, 2026. Download files separately; the scripts do not download data automatically.

| Input                       | What to download                                                                                                                                                                                                                               | Destination under the extracted package                                                                                          |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| Highway network             | [FAF5 Model Highway Network ZIP](https://ops.fhwa.dot.gov/freight/freight_analysis/faf/faf_highway_assignment_results/FAF5_Model_Highway_Network.zip), linked from the [FHWA FAF page](https://ops.fhwa.dot.gov/freight/freight_analysis/faf/) | Preserve the full extracted folder, including `data/raw/FAF5_Model_Highway_Network/Networks/Geodatabase Format/FAF5Network.gdb/` |
| Regional demand             | Regional CSV database containing the 2017 baseline and mid-range forecasts, from the [BTS FAF5 page](https://www.bts.gov/faf/faf5); verified current link: [FAF5.7.1.zip](https://faf.ornl.gov/faf5/data/download_files/FAF5.7.1.zip)          | Extract `data/raw/FAF5.7.1.csv`                                                                                                  |
| County factors              | [All experimental county disaggregation factors ZIP](https://faf.ornl.gov/faf5/Data/County/All_Experimental_Disaggregation_Factors.zip), from the [BTS county page](https://www.bts.gov/faf/county)                                            | `data/raw/All_Experimental_Disaggregation_Factors/truck_origin_factors.csv` and `truck_destination_factors.csv`                  |
| STRAHNET attributes         | GeoPackage export from [BTS Strategic Highway Network](https://geodata.bts.gov/datasets/usdot::strategic-highway-network-strahnet/about)                                                                                                       | Rename to `data/raw/NTAD/NTAD_Strategic_Highway_Network.gpkg`                                                                    |
| NHS attributes              | GeoPackage export from [BTS National Highway System](https://geodata.bts.gov/datasets/usdot::national-highway-system-nhs/about)                                                                                                                | Rename to `data/raw/NTAD/NTAD_National_Highway_System.gpkg`                                                                      |
| National Network attributes | GeoPackage export from [BTS National Network](https://geodata.bts.gov/datasets/usdot::national-network/about)                                                                                                                                  | Rename to `data/raw/NTAD/NTAD_National_Network.gpkg`                                                                             |

For the three NTAD pages, choose **Download → GeoPackage** and retain the attributes. These layers supply the matched AADT/background-traffic capacity treatment. Omitting them would change that treatment. Their release vintages and file signatures are recorded by validation. If an export has multiple layers, set its explicit layer name in `legacy_options.ntad` in the upstream config. County boundary shapefiles are unnecessary: county FIPS and centroids come from the FAF network.

Expected raw directory:

```text
data/raw/
  FAF5.7.1.csv
  FAF5_Model_Highway_Network/
    Networks/Geodatabase Format/FAF5Network.gdb/
  All_Experimental_Disaggregation_Factors/
    truck_origin_factors.csv
    truck_destination_factors.csv
  NTAD/
    NTAD_Strategic_Highway_Network.gpkg
    NTAD_National_Highway_System.gpkg
    NTAD_National_Network.gpkg
```

**Version choice is explicit.** The manuscript used FAF5.6.1, while the verified current FAF5 CSV link supplies 5.7.1. The default `configs/reproduce.json`/`configs/upstream.json` use the public 5.7.1 release's `tons_2017`. They do not switch to FAF6. The county factors were published against FAF5.6.1; see [their documentation](https://www.bts.gov/faf/county/documentation). If you already have the manuscript's original demand release, place `FAF5.6.1.csv` under `data/raw/` and use `configs/reproduce_manuscript561.json`. An archived 5.6.1 download could not be verified, so no purported archive URL is supplied.

Demand and NTAD vintage differences can change recalculated costs, traces, models and results. The frozen recipe requires the same highway-network stable-ID universe and ordering. A different network edition or unmatched cleaned IDs stops with an explicit error; the code does not invent mappings. Keep the FAF5 network data dictionary with your downloaded inputs.

## 3. Run the declared workflow

The recommended supervisor records phase, step, run ID, configuration/code hashes, commands, outputs and resource use. A run always requires an explicit configuration, run ID and stage list.

First inspect the full command plan without computation:

```text
python scripts/run_pipeline.py --config configs/reproduce.json --run-id faf2017_v1 --all --dry-run
```

To execute all 18 declared stages, including the final sample-fraction and augmented-data experiments:

```text
python scripts/run_pipeline.py --config configs/reproduce.json --run-id faf2017_v1 --all
```

For the original FAF5.6.1 profile, replace the config argument with `configs/reproduce_manuscript561.json` throughout. Use a fresh extraction/output workspace when switching profiles.

For step-by-step execution, use the following commands in order. Stop after `evaluate` if you need the main national workflow without the separate Figure 5 computational experiments.

```text
python scripts/run_pipeline.py --config configs/reproduce.json --run-id faf2017_v1 --stage validate-inputs
python scripts/run_pipeline.py --config configs/reproduce.json --run-id faf2017_v1 --stage clean-network
python scripts/run_pipeline.py --config configs/reproduce.json --run-id faf2017_v1 --stage disaggregate-demand
python scripts/run_pipeline.py --config configs/reproduce.json --run-id faf2017_v1 --stage enrich-network
python scripts/run_pipeline.py --config configs/reproduce.json --run-id faf2017_v1 --stage snapshots
python scripts/run_pipeline.py --config configs/reproduce.json --run-id faf2017_v1 --stage physical
python scripts/run_pipeline.py --config configs/reproduce.json --run-id faf2017_v1 --stage consolidate
python scripts/run_pipeline.py --config configs/reproduce.json --run-id faf2017_v1 --stage topology
python scripts/run_pipeline.py --config configs/reproduce.json --run-id faf2017_v1 --stage references
python scripts/run_pipeline.py --config configs/reproduce.json --run-id faf2017_v1 --stage support
python scripts/run_pipeline.py --config configs/reproduce.json --run-id faf2017_v1 --stage labels
python scripts/run_pipeline.py --config configs/reproduce.json --run-id faf2017_v1 --stage retry-labels
python scripts/run_pipeline.py --config configs/reproduce.json --run-id faf2017_v1 --stage train
python scripts/run_pipeline.py --config configs/reproduce.json --run-id faf2017_v1 --stage freeze
python scripts/run_pipeline.py --config configs/reproduce.json --run-id faf2017_v1 --stage assign
python scripts/run_pipeline.py --config configs/reproduce.json --run-id faf2017_v1 --stage evaluate
python scripts/run_pipeline.py --config configs/reproduce.json --run-id faf2017_v1 --stage sample-fractions
python scripts/run_pipeline.py --config configs/reproduce.json --run-id faf2017_v1 --stage enrichment
```

## 4. What the stages compute

| Stage                     | Computation and main output                                                                                                                                                                                         |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `validate-inputs`         | Verify network layers, demand/factor schemas, nonempty features and downloaded-file identities.                                                                                                                     |
| `clean-network`           | Exact duplicate cleaning, 100 m endpoint resolution in EPSG:5070, lane-directed arcs, declared missing-lane treatment, stable node/link/arc IDs; clean GeoPackage and SQLite.                                       |
| `disaggregate-demand`     | Truck `dms_mode=1`, 2017 annual thousand tons, five commodity groups, published origin/destination county factors, equal division among each county's FAF centroids; SQL-batched centroid OD and demand accounting. |
| `enrich-network`          | Original three-layer geographic matching and capacity/FFTT formulas, with array evaluation replacing scalar per-arc capacity loops; enriched GIS and SQLite.                                                        |
| `snapshots`               | Dense stable-ID ordered directed graph and OD NPZ/Parquet arrays, metadata and `upstream/receipt.json`. Empty inputs, missing IDs or silently dropped physical arcs fail.                                           |
| `physical`                | Original physical graph safeguarded CFW, first fresh LOW gap ≤1e-3, continuation to HIGH update 200, checkpoints, separated CURRENT/AON physical and s03 traces.                                                    |
| `consolidate`             | Apply the fixed split/B quotient/bundle/macro recipe; regenerate LOW-based static B charges and certified explicit power4 backbone functions.                                                                       |
| `topology`                | Assemble the common P physical-residual network and mapped OD; fixed s03 partition plus four newly generated fixed-backbone weighted-Leiden partitions.                                                             |
| `references`              | Fresh OPEN and represented-I95 P CFW assignments, requested relative gap 1e-4, complete served-demand membership, witnesses and all five trace families.                                                            |
| `support`                 | Freeze supported regional movements, build regional detailed graphs and abstract graph arrays, audit physical witnesses and shared interfaces.                                                                      |
| `labels` / `retry-labels` | Freeze unique TRAIN/VALID/TEST requests for active s03 regions; solve/certify physical regional labels and preserve unavailable attempts. Other resolutions use FFTT without learned training.                      |
| `train`                   | Fresh s03 FFTT, Power4, ICNN and XGBoost models, finite TRAIN/VALID metrics, native model exports and numerical-eligibility audits.                                                                                 |
| `freeze`                  | Persist whole-region choices before TEST: retained-model ICNN policy and accuracy-qualified XGBoost policy, plus a separate accuracy-qualified comparison portfolio.                                                |
| `assign`                  | Sixteen OPEN/I95 query dispositions: four s03 families and FFTT on the other four resolutions; fresh certificates, matched P served membership and flow arrays.                                                     |
| `evaluate`                | Held-out regional cost metrics after frozen selection, with missing labels and model availability recorded.                                                                                                         |
| `sample-fractions`        | Sixteen frozen manuscript region slots × five nested TRAIN fractions × two families: 160 independent fits, then common held-out evaluation.                                                                         |
| `enrichment`              | Thirty-two additional requested TRAIN inputs per selected region, certified admission, separate full augmented fits for both families: 32 fits. VALID/TEST and national portfolios remain fixed.                    |

The dependency order is cleaned inputs → original physical assignment → numerical B consolidation → common P/five partitions → P references/support → regional labels/models → abstract assignment. The selected s03 split already exists as the supplied mapping recipe so the original physical assignment can be traced consistently. The four additional fixed-B partitions are generated after consolidation. This preserves the final method without replaying all historic phase searches.

### Cost, flow and topology conventions

Raw demand and solver flows are **annual thousand tons**, not vehicles. Graph-snapshot free times are minutes and `length_miles` is in miles; GIS also stores meter-based lengths and EPSG:5070 geometry. The common P/abstract cost coefficients use seconds. Source/base freight capacity, prepared assignment capacity and scenario capacity are distinct. Preparation applies 2/3 to ordinary arcs and replaces that factor with 100 for centroid-connector class 50. The physical/P scenario then applies 0.30 to prepared capacity exactly once.

CURRENT is the safeguarded-CFW convex-mixture flow. Latest AON is the auxiliary shortest-path flow at CURRENT costs. Only LOW CURRENT calibrates operating-flow charges and backbone coefficients; LOW AON contributes declared fit-domain evidence. HIGH is evaluation evidence. The new LOW state need not be iteration 45. HIGH stops at 200 updates and does not assert the original 1e-4 target was achieved.

B charges use fresh suppressed physical exposure `E=sum(flow*time_seconds)` divided by half incoming-plus-outgoing static incidence `V`; when `V=0`, use the frozen rule's fresh internal median FFTT fallback. Charges are unclipped. Macro internal charges occur once and endpoints receive half charges. Exact primitive/series functions and explicit physical members retain their native formulas; heterogeneous selected macros use the native LOW-only offline oracle fit and independent ≤1% cost/potential certification. A rejected fit preserves its evidence and stops, rather than silently retuning or retaining an old coefficient.

The P comparison network retains the consolidated backbone, residual detailed physical multiarcs and separate nontransit O/D role connectors. It is distinct from the original physical graph benchmark. All comparison networks preserve common backbone arc IDs/order/costs and mapped OD. Represented I95 closure disables the frozen represented backbone coordinates while leaving residual physical alternatives available. Unreachable demand is accounted for and never renormalized; within each scenario, P and surrogate assignments must serve the same OD membership.

Primary resolution is `s03` (`r=1`). The other fixed-B resolutions are `s02=0.3107232505953859`, `s04=1.6`, `s05=4`, `s06=12`; `s01` is excluded. s03 reuses the final directed-arc mapping. The other four use weighted Leiden, seed 42, modularity, beta 0.01 and the original residual ownership rules, with explicit singleton handling for directed residual B-to-B arcs. The code preserves multiarcs and distinct centroid roles.

### Label and model recipes

Default fresh s03 requests are 64 TRAIN, 24 VALID and 48 TEST inputs per supported region, with immutable role/index identities. Certified CFW labeling has 20-second and 60-second attempts, at most 20,000 updates per attempt, best certified state retention and explicit absent labels. The standalone schedule omits obsolete intermediate data expansions; historical training pools and allocation paths need not recur. Fresh national ICNN fitting uses widths 32/64/128, float64 Adam, 250-update blocks and the declared learning-rate ladder. Power4 fits nonnegative coefficients in physical cost units; XGBoost fits centered log costs and maps them back by exponentiation. All settings and resource caps are explicit in `configs/downstream.json`.

ICNN costs remain gradients of the anchored two-layer Softplus scalar potential with a nonnegative Power4 branch and neural amplitude. At zero neural amplitude the model is Power4-equivalent; the code does not invent a positive activity floor. Numerical eligibility, TRAIN error below 1% and held-out accuracy are separate records. TRAIN/VALID select models; TEST cannot select or retrain them.

`PORTFOLIOS.json` uses numerically eligible retained ICNNs even when TRAIN error exceeds 1%, with fixed whole-region analytic fallback for unavailable/inconclusive components. `PORTFOLIOS_ACCURACY_QUALIFIED.json` separately requires learned components to meet the strict TRAIN target. Default XGBoost uses the accuracy-qualified policy. These distinguish the manuscript's selected retained ICNN assignment from its separate refitted/accuracy-qualified comparison. Fresh candidates and portfolio counts can differ from the historical selected models. After the default matrix, optional alternate ICNN assignments can be invoked with `python scripts/downstream/run_downstream.py assign-qualified --config configs/downstream.json --run-id faf2017_v1` while the downstream clock still permits it. Replace `faf2017_v1` with the same run ID used above. This direct optional action uses the adapter's resource guards; the supervisor's separate campaign clock and 150 GB output guard do not cover it.

The last two experiment stages reuse the latest manuscript's fitting functions unchanged: width 32, 1,000 Adam updates, audits every 250 updates, batch size at most 128; XGBoost 32 rounds, depth 3, learning rate 0.05. Fractions are 10%, 25%, 50%, 70%, 100%, using seeded nested prefixes. The frozen 16 region slots/strata are in `configs/downstream_figure5_regions.json`. Missing slots in fresh support stop explicitly; slots are not renumbered to improve results. Added TRAIN inputs use seed 2026093001 and are checked against every prior role. Selection freezes before TEST evaluation. These independent fits do not replace the national portfolios.

## 5. Locate and check generated outputs

Everything generated is under `data/processed/` or `runs/`:

| Directory                                                              | Contents                                                                                                                                                                                                                              |
| ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `data/processed/upstream/partC_rev1_cache/`                            | Clean/enriched GeoPackages and SQLite network/centroid OD tables; open the GeoPackages in QGIS if desired.                                                                                                                            |
| `data/processed/upstream/snapshots/`                                   | Graph/OD NPZ, node/arc Parquets and metadata. `receipt.json` in the upstream root connects exact paths/identities to middle stages.                                                                                                   |
| `data/processed/middle/derived_inputs/`                                | Exact duplicate-endpoint OD aggregation, original multiplicities and provenance.                                                                                                                                                      |
| `data/processed/middle/physical/`                                      | LOW/HIGH flow/trace arrays, native checkpoints, iteration history and fresh metrics.                                                                                                                                                  |
| `data/processed/middle/consolidation/`                                 | Regenerated backbone links/coefficients, static charges and offline fit evidence.                                                                                                                                                     |
| `data/processed/middle/topology/`                                      | Common P graph/OD, five partition arrays, interfaces, ownership and identity report.                                                                                                                                                  |
| `data/processed/middle/references/`                                    | OPEN/I95 P references, witnesses, coverage and traces.                                                                                                                                                                                |
| `data/processed/middle/regions/`, `acnhn/`, `training/`, `portfolios/` | Regional detailed graphs, abstract arrays, immutable label requests/attempts, models/audits and fixed selection registries. `SUPPORT_FREEZE.json` and `COMPLETE_WITNESS_AUDIT.json` in the middle root bind support/witness evidence. |
| `data/processed/middle/assignments/`, `evaluation/`                    | Abstract query results/certificates and post-freeze TEST metrics.                                                                                                                                                                     |
| `data/processed/middle/experiments/`                                   | Separate sample-fraction/augmented models, request dispositions, selection freezes and TEST evaluations.                                                                                                                              |
| `data/processed/middle/reproduction_logs/`                             | Per-action intents, input manifests and domain result/resource receipts.                                                                                                                                                              |
| `runs/<run-id>/`                                                       | Supervisor commands/logs, artifact hashes, resource samples and stage results.                                                                                                                                                        |

Inspect the adapter's domain result along with the supervisor's receipt. A supervisor PASS means its declared task produced the required nonempty artifacts; it is not a claim that every model met 1%, every assignment converged or every label was available. P references must meet their requested gap before support construction. Surrogate assignment time limits, unresolved numerical checks and unavailable labels remain explicit rather than becoming zero errors. Empty graphs, OD, support or outputs are failures.

# 
