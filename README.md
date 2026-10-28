# PumpWatch

An end-to-end implementation of **Detection of Pump-and-Dump Schemes in Cryptocurrency Markets Using Machine Learning**.

Includes real historical data, four trained classifiers, chronological evaluation, a Streamlit dashboard, explanations, historical dataset evaluation, historical replay, and reproducible research artifacts. It detects windows similar to documented pump onsets; it does not establish intent or predict a pump before it starts.

## Start the application

The project was built and tested with Python 3.13. The included models require the pinned package versions, especially scikit-learn.

On your Mac, open Terminal, change into the extracted `pumpwatch` folder, and run:

```bash
python3.13 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m streamlit run app.py --server.address 127.0.0.1
```

Then visit **http://127.0.0.1:8501**. If `python3.13` is not found, install Python 3.13 first. The included trained models use the pinned dependencies.

On Windows, use PowerShell from the project folder:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m streamlit run app.py
```

Alternatively run `setup.ps1`, followed by `run.ps1`. Installation needs internet access. Once installed, the shipped data, models and dashboard work offline. Models are trusted local project artifacts; never load an unknown joblib/pickle file.

## Dashboard

| Screen | Function |
|---|---|
| Overview | Dataset coverage, held-out metrics and navigation |
| Historical explorer | Real candles, alerts and local explanations |
| Model comparison | Four models, baselines, confusion matrices, PR curves, event recall and errors |
| Historical replay | Reveal a candle prefix and run inference only on visible data |
| Data & methodology | Label policy, splits, limitations, audit and downloads |

Risk scores lie between 0 and 1 but are **not calibrated probabilities**. The threshold is chosen using validation F1. The SVM decision margin is transformed monotonically to this range; it is not probability-calibrated. Negative downsampling and event-conditioned sampling further limit probability interpretation. `artifacts/score_reliability.csv` compares validation score bins with observed labeled prevalence.

## Historical-data scope

The dashboard works only with the project's historical dataset and saved train/validation/test results. It has no file-upload page or live data feed. Historical explorer and replay are retained to explain predictions on the existing dataset.

The small files in `examples/` are historical fixtures used by the software tests, not an input workflow. The existing OHLCV-only experiment remains available as a comparison with the trade-activity model; its results and threshold have not changed.

## Edit dashboard wording

Edit `pumpwatch/dashboard_text.py` for page titles, sidebar names, overview copy and the method/result explanations. Keep dictionary keys unchanged; change only their quoted values. For detailed examples, see `EDITING_DASHBOARD.md`.

Other widget labels and chart captions are in `app.py`. Theme colors are in `.streamlit/config.toml`, with additional layout styles near the top of `app.py`. Wording changes do not require retraining.

## Reproduce the research

From the project folder on a Mac with the virtual environment installed:

```bash
./.venv/bin/python -m pumpwatch.data
./.venv/bin/python -m pumpwatch.train
./.venv/bin/python -m pumpwatch.reporting
./.venv/bin/python -m unittest discover -s tests -v
./.venv/bin/python tests/check_dashboard.py
```

Run from this project directory. Acquisition requires network access and reuses cached source files. Retraining overwrites generated model/evaluation artifacts; it does not modify original source files. Acquisition keeps a manifest of failed archives and verifies successful archives against the published checksums.

The included notebook has already been executed successfully. To execute it again in a notebook-capable editor, install `requirements-notebook.txt` and select that Python environment. The standalone HTML report opens offline without a notebook environment.

1. Research labels: extract positive timestamps from the authors' `features_25S.csv.gz`; match each to the same coin's Binance announcement within five minutes. The source feature columns never enter our classifiers.
2. Download the corresponding coin/BTC event day and the same pair seven days earlier. These are full UTC days, with their first 60 minutes unavailable for prediction unless continuous preceding data is present.
3. Compute returns, acceleration, preceding-window volume anomaly, volatility, candle shape and trade-activity features. No future data or event identifiers are model inputs.
4. Label onset minute plus the next two minutes positive. Exclude all other minutes within ±60 minutes of any known onset or announcement. Negative means no known event.
5. Split chronologically at approximately 60% and 80% of unique event dates. Purge 60 minutes of feature history and the target duration at boundaries. Events cannot straddle partitions.
6. Fit on all eligible training positives and up to 10,000 sampled negatives. Scaling and weights are learned on training only. Validation and test retain eligible negatives.
7. Compare two settings each for Logistic Regression, Random Forest, RBF SVM and histogram-based Gradient Boosting. Select settings and thresholds by validation F1. This is a bounded single-holdout experiment, not nested cross-validation.
8. Freeze model choice, then compute held-out results and event metrics. Average precision is the stepwise PR summary, not trapezoidal PR-AUC.
9. Fit an additional OHLCV-only version of the selected algorithm to compare historical evaluation without trade activity. Give it its own validation threshold and test evaluation.

## Evaluation and explanations

- `artifacts/report.json`: full metrics, chosen parameters, split dates and counts, unseen-coin subset, fallback results and environment versions.
- `artifacts/model_comparison.csv`: main held-out comparison.
- `artifacts/event_results.csv`: independent test-event outcomes and detection delays.
- `artifacts/errors.csv`: false-positive and false-negative windows.
- `artifacts/event_coverage.csv`: which documented events have complete usable target windows.
- `artifacts/importance.csv`: validation permutation importance by average-precision decrease.
- `artifacts/score_reliability.csv`: validation score bins and observed prevalence.

Event recall counts events with an alert in one of their three positive minutes. Detection delay uses the completed candle's close time and the source's annotated onset. False-alert episodes collapse consecutive alert minutes. Rates use eligible negative exposure on the collected days; they are not a market-wide deployment estimate. Ambiguous intervals are excluded from both the false-alert numerator and exposure denominator.

Local explanations replace one feature at a time with its training median and measure the score change. These are **non-additive sensitivity measurements**, not SHAP values or causal evidence. Correlated features can make median substitution unrealistic. Global permutation importance is also descriptive, not causal.

## Project files

```text
app.py                       Local dashboard
pumpwatch/dashboard_text.py  Editable dashboard headings and explanations
pumpwatch/data.py            Acquisition, integrity and input validation
pumpwatch/features.py        Causal features and labels
pumpwatch/train.py           Splits, models, thresholds and evaluation
pumpwatch/inference.py       Shared prediction and explanation functions
pumpwatch/reporting.py       Report, audit tables, notebook and samples
data/raw/                    Original downloaded sources and checksums
data/candles.csv.gz          Cleaned real market candles
artifacts/                   Trained models, predictions and evaluation
examples/                    Real CSV demonstration data
tests/                       Causality, validation, splitting and integration checks
Exploratory_Analysis.ipynb    Re-runnable analysis notebook
Research_Report.html         Standalone offline report with interactive charts
```

## Demonstration walkthrough

1. Open Overview and explain that model choice was made on validation data.
2. Open Model comparison; show test precision, recall, F1, the confusion matrix and false alerts.
3. Open Historical explorer and keep the held-out test collection selected. Inspect price, volume and the three-minute detection window.
4. Select a candle and explain the score using the sensitivity chart and actual feature values.
5. Open Historical replay. Advance the slider through the event; each step uses only the revealed prefix.
6. Finish with Data & methodology and discuss the limitations below.

## Limitations and intended scope

This is a historical academic research application. It does not execute trades or provide a live feed. Data ends in January 2021; contemporary generalization is not established. Sampling around known events and matched days overrepresents relevant markets. Unrecorded pumps can contaminate negatives. Manual onsets have timing uncertainty, and one-minute candles hide faster activity. A three-minute positive window differs from the original paper's exact evaluation. Order-book anomalies and Telegram ingestion are not implemented. There are no deployment claims or confidence intervals. Inspect event counts as well as window counts.

The eight missing daily files concern YOYOWBTC and are explicitly listed in the manifest. Do not silently replace BTC-pair event data with USDT pairs. Optional extensions are live monitoring, broader negative-market coverage, temporal cross-validation, probability calibration on representative data, interval-sensitivity experiments and finer trade-level data.

## Sources and attribution

La Morgia, M., Mei, A., Sassi, F., and Stefa, J. (2020). *Pump and Dumps in the Bitcoin Era: Real Time Detection of Cryptocurrency Market Manipulations*. ICCCN. DOI: 10.1109/ICCCN49398.2020.9209660.

- Research source: https://github.com/SystemsLab-Sapienza/pump-and-dump-dataset
- Original source license is preserved at `data/raw/LICENSE`.
- Market archive documentation: https://github.com/binance/binance-public-data
- Market data terms: https://github.com/binance/binance-public-data/blob/master/TERMS_AND_CONDITIONS.md
- Research source hashes: `data/audit.json`; exchange URLs/checksums: `data/download_manifest.csv`.

The implementation is a new one-minute OHLCV experiment informed by the supplied synopsis; it does not claim to reproduce the original paper's results.
