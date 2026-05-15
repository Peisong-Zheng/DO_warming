# DO-like Asian monsoon transition timing

This repository contains the analysis code for a manuscript on the timing of
DO-like Asian monsoon transitions. The central question is whether transition
times identified in the Chinese speleothem composite are organized by slow
glacial background state and orbital phase.

The main event catalogue is from Rousseau et al. (2023), who used a KS-window
method to identify weak- and strong-monsoon starts in the 0--640 kyr Cheng et
al. (2016) Chinese speleothem composite. This project treats those transition
ages as an event sequence. The analyses ask whether:

- events cluster at preferred precession or obliquity phases;
- LR04 and CO2 improve event-rate prediction as slow climate-state predictors;
- precession phase adds information after event history, sampling resolution,
  LR04, and CO2 are included;
- the results are robust to bin width, age uncertainty, KS-window choice,
  additional forcings, bootstrap LR nulls, and an independent Barker et al.
  (2011) D-O warming catalogue.

## Layout

```text
data/
  raw/          Input data. Scripts read from this directory but do not write to it.
  processed/    Script-generated CSV outputs.

figures/        Script-generated figures, grouped by script name.

monsoon_paper/
  main.tex      Main manuscript.
  SI.tex        Supporting Information.
  reference.bib Reference library.
  figures/      Flat Overleaf-ready figure directory: Fig01.pdf, FigS01.pdf, ...

toolbox/        Shared statistical and likelihood utilities.

run_logs/       Timestamped logs from workflow and validation runs.
```

The intended convention is that `data/raw/` is input-only. Generated tables and
figures are written under `data/processed/<script_name>/` and
`figures/<script_name>/`.

## Main Workflow

To regenerate the main-text analyses without running sensitivity experiments:

```bash
python run_main_paper_workflow.py
```

This runs:

1. `Predictive_information_model.py`
2. `Orbital_phase_rayleigh.py`
3. `Lagged_predictive_information.py`
4. `paper_figure_export.py`

The first three scripts regenerate the main scientific results. The final
script copies only manuscript-referenced PDF figures into
`monsoon_paper/figures/`, using names such as `Fig01.pdf` and `FigS01.pdf`.

To inspect the commands without running them:

```bash
python run_main_paper_workflow.py --dry-run
```

To skip the final figure-copy step:

```bash
python run_main_paper_workflow.py --skip-export
```

Workflow logs are written to `run_logs/main_paper_workflow/`.

## Model Names

The manuscript and code use three recurring model names.

- **Event-process baseline**: same-type event history plus local sampling
  resolution of the Cheng composite. This is a nuisance-control model for
  event clustering and uneven sampling, not a climate-forcing model.

- **Climate-state model**: event-process baseline plus LR04 and CO2.

- **Full predictive model**: climate-state model plus precession phase,
  represented by sine and cosine terms.

The term **extended baseline** is used only in the additional-forcing
sensitivity experiment. There it denotes the full predictive model before
adding Antarctic temperature, obliquity, eccentricity, or 65N summer-solstice
insolation.

## Main Scripts

| Script | Purpose | Main outputs |
|---|---|---|
| `Predictive_information_model.py` | Fits the event-process baseline, climate-state model, and full predictive model for Rousseau weak/strong monsoon starts. It also writes the main fitted-rate and model-comparison figures. | `data/processed/Predictive_information_model/`; `figures/Predictive_information_model/`; manuscript Fig01 and Fig03 |
| `Orbital_phase_rayleigh.py` | Converts precession and obliquity to phase variables and tests event-phase clustering with Rayleigh tests. | `data/processed/Orbital_phase_rayleigh/`; `figures/Orbital_phase_rayleigh/`; manuscript Fig02, FigS01, and FigS02 |
| `Lagged_predictive_information.py` | Scans lagged predictive-information gains. LR04 and CO2 are added separately to the event-process baseline; lagged precession phase is added to the climate-state model. | `data/processed/Lagged_predictive_information/`; `figures/Lagged_predictive_information/`; manuscript Fig04 |
| `Bin_hazard_phase_poisson.py` | Earlier binned Poisson hazard model without event-process controls. It is retained for one SI comparison figure showing LR04+CO2 and LR04+CO2+precession fitted rates without the noisier event-process baseline. | `data/processed/Bin_hazard_phase_poisson/`; `figures/Bin_hazard_phase_poisson/`; manuscript FigS03 |
| `paper_figure_export.py` | Copies only paper-referenced PDF figures from `figures/` to `monsoon_paper/figures/`. Plotting scripts also call its helper functions when saving manuscript figures. | `monsoon_paper/figures/Fig*.pdf` |

## Sensitivity and Diagnostic Scripts

These scripts are not run by `run_main_paper_workflow.py`.

| Script | Purpose |
|---|---|
| `Bin_hazard_phase_poisson_sensitivity.py` | Tests whether Antarctic temperature, obliquity, eccentricity, and 65N summer-solstice insolation add information beyond the full predictive model. |
| `Bin_hazard_phase_poisson_binwidth_sensitivity.py` | Repeats the core predictive-model comparisons for bin widths from 0.2 to 1.0 kyr. |
| `Composite_age_uncertainty_core_sensitivity.py` | Perturbs event ages using a conservative Cheng composite age-uncertainty envelope and repeats Rayleigh and predictive-information checks. |
| `KS_window_core_experiment_sensitivity.py` | Compares Rousseau 0.4--4 kyr and 0.6--4 kyr KS-window catalogues. |
| `Predictive_information_bootstrap_diagnostics.py` | Runs reduced-model parametric bootstrap null tests for the two central LR comparisons. This is slower than the other diagnostics; the default is 2000 bootstrap replicates. |
| `Barker2011_do_predictive_information.py` | Repeats Rayleigh and predictive-information analyses for the Barker et al. (2011) variable-threshold D-O warming catalogue. |

## Shared Code

`toolbox/` contains small shared utilities:

- `toolbox/poisson.py`: binned Poisson log likelihood, optimization objective,
  design matrix construction, and fitted-rate conversion.
- `toolbox/model_stats.py`: AIC, AICc, BIC, likelihood-ratio p values,
  log-likelihood gains, and bits/event summaries.

Plotting code remains inside the individual analysis scripts. Several figures
share a broad structure but differ in small, hand-tuned details, so keeping the
plotting local is less fragile than forcing a single plotting abstraction.

## Dependencies

The scripts use the standard scientific Python stack:

```text
numpy
pandas
scipy
matplotlib
xarray
openpyxl
xlrd
```

Some Excel files may produce an `openpyxl` warning about unsupported workbook
extensions. This comes from workbook metadata and does not affect the numerical
outputs.

## Main Data Sources

The entries below are copied from `monsoon_paper/reference.bib` for quick
orientation.

- Barker, S., Knorr, G., Edwards, R. L., Parrenin, F., Putnam, A. E.,
  Skinner, L. C., Wolff, E., & Ziegler, M. (2011).
  800,000 years of abrupt climate variability. *Science*, 334(6054), 347--351.

- Bereiter, B., Eggleston, S., Schmitt, J., Nehrbass-Ahles, C., Stocker, T. F.,
  Fischer, H., Kipfstuhl, S., & Chappellaz, J. (2015).
  Revision of the EPICA Dome C CO2 record from 800 to 600 kyr before present.
  *Geophysical Research Letters*, 42(2), 542--549.

- Cheng, H., Edwards, R. L., Sinha, A., Spötl, C., Yi, L., Chen, S.,
  Kelly, M., Kathayat, G., Wang, X., Li, X., et al. (2016).
  The Asian monsoon over the past 640,000 years and ice age terminations.
  *Nature*, 534(7609), 640--646.

- Corrick, E. C., Drysdale, R. N., Hellstrom, J. C., Capron, E.,
  Rasmussen, S. O., Zhang, X., Fleitmann, D., Couchoud, I., & Wolff, E. (2020).
  Synchronous timing of abrupt climate changes during the last glacial period.
  *Science*, 369(6506), 963--969.

- Laskar, J., Robutel, P., Joutel, F., Gastineau, M., Correia, A. C. M.,
  & Levrard, B. (2004).
  A long-term numerical solution for the insolation quantities of the Earth.
  *Astronomy & Astrophysics*, 428(1), 261--285.

- Lisiecki, L. E., & Raymo, M. E. (2005).
  A Pliocene-Pleistocene stack of 57 globally distributed benthic δ18O records.
  *Paleoceanography*, 20(1).

- Lohmann, J., & Ditlevsen, P. D. (2018).
  Random and externally controlled occurrence of Dansgaard-Oeschger events.
  *Climate of the Past*, 14(5), 609--617. https://doi.org/10.5194/cp-14-609-2018

- Rousseau, D.-D., Bagniewski, W., & Cheng, H. (2023).
  A reliable benchmark of the last 640,000 years millennial climate variability.
  *Scientific Reports*, 13(1), 22851.
