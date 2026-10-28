"""Generate an offline research report, exploratory notebook, and demo samples."""
from __future__ import annotations
import hashlib
import html
import json
import importlib.metadata

import joblib
import nbformat as nbf
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from sklearn.base import clone
from sklearn.utils.class_weight import compute_sample_weight

from .data import ROOT, DATA
from .features import FEATURES
from .train import ARTIFACTS, SEED, scores, choose_threshold, metrics


def generate():
    report=json.loads((ARTIFACTS/'report.json').read_text())
    audit=json.loads((DATA/'audit.json').read_text())
    d=pd.read_csv(ARTIFACTS/'predictions.csv.gz')
    d['timestamp']=pd.to_datetime(d.timestamp,utc=True)
    d['prediction_time']=pd.to_datetime(d.prediction_time,utc=True)
    bundle=joblib.load(ARTIFACTS/'best_model.joblib')
    # Support existing full-model runs as well as one-command retraining.
    if 'core_model' not in bundle:
        train=d[(d.split=='train')&(d.target>=0)]
        fit=pd.concat([train[train.target==1], train[train.target==0].sample(min(10000,int((train.target==0).sum())),random_state=SEED)]).sample(frac=1,random_state=SEED)
        val=d[(d.split=='validation')&(d.target>=0)]
        test=d[(d.split=='test')&(d.target>=0)]
        names=[n for n in FEATURES if n not in ['trade_z','mean_trade_size','buy_share','activity_available']]
        model=clone(bundle['model'])
        kwargs={'sample_weight':compute_sample_weight('balanced',fit.target)} if bundle['name']=='Gradient Boosting' else {}
        model.fit(fit[names],fit.target,**kwargs)
        threshold=choose_threshold(val.target,scores(model,val[names]))
        bundle.update(core_model=model,core_features=names,core_threshold=threshold)
        joblib.dump(bundle,ARTIFACTS/'best_model.joblib')
        report['ohlcv_only_test']=metrics(test.target,scores(model,test[names]),threshold)
    events=pd.read_csv(DATA/'events.csv')
    coverage=d[d.target==1].groupby('event_id').agg(eligible_minutes=('target','size'),split=('split','first')).reset_index()
    coverage=events[['event_id','pair','onset']].merge(coverage,on='event_id',how='left')
    coverage['eligible_minutes']=coverage.eligible_minutes.fillna(0).astype(int)
    coverage['status']=np.where(coverage.eligible_minutes==3,'complete',np.where(coverage.eligible_minutes>0,'partial','unavailable or not eligible'))
    coverage.to_csv(ARTIFACTS/'event_coverage.csv',index=False)
    report['event_coverage']={'documented':len(events),'complete':int((coverage.status=='complete').sum()),
                              'partial':int((coverage.status=='partial').sum()),'unavailable':int((coverage.eligible_minutes==0).sum())}
    versions={n:importlib.metadata.version(n) for n in ['numpy','pandas','scikit-learn','streamlit','plotly','requests','joblib','nbformat']}
    report['environment']=versions
    audit['research_file_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (DATA/'raw').iterdir() if p.is_file()}
    audit['usable_event_coverage']=report['event_coverage']
    (DATA/'audit.json').write_text(json.dumps(audit,indent=2))
    (ARTIFACTS/'report.json').write_text(json.dumps(report,indent=2))

    # Validation-score bins are descriptive: displayed score is not a probability.
    val=d[(d.split=='validation')&(d.target>=0)].copy()
    val['score_bin']=pd.cut(val.risk_score,bins=np.linspace(0,1,11),include_lowest=True)
    reliability=val.groupby('score_bin',observed=True).agg(rows=('target','size'),mean_score=('risk_score','mean'),observed_pump_fraction=('target','mean'))
    reliability.to_csv(ARTIFACTS/'score_reliability.csv')
    frame=pd.DataFrame([{ 'Model':n,**{k:v['test'][k] for k in ['precision','recall','f1','average_precision','roc_auc']}} for n,v in report['models'].items()])
    frame.to_csv(ARTIFACTS/'model_comparison.csv',index=False)
    test=d[(d.split=='test')&(d.target>=0)]
    errors=test[(test.alert & (test.target==0)) | (~test.alert & (test.target==1))].copy()
    errors['error_type']=np.where(errors.target==0,'false positive','false negative')
    errors[['timestamp','symbol','target','risk_score','event_id','error_type']].to_csv(ARTIFACTS/'errors.csv',index=False)
    summary=d[d.target>=0].groupby(['split','target']).agg(rows=('target','size'),median_return_1m=('return_1m','median'),median_volume_z=('volume_z','median'),median_trade_z=('trade_z','median'))
    summary.to_csv(ARTIFACTS/'eda_summary.csv')
    figures=[]
    fig=px.bar(frame,x='Model',y=['precision','recall','f1'],barmode='group',title='Held-out test comparison',color_discrete_sequence=['#22a38b','#477ec5','#e2a337'])
    figures.append(fig)
    sample=pd.concat([d[(d.split=='train')&(d.target==1)],d[(d.split=='train')&(d.target==0)].sample(3000,random_state=SEED)])
    sample['Class']=sample.target.map({0:'No known event',1:'Pump-onset window'})
    fig2=px.box(sample,x='Class',y='volume_z',color='Class',title='Training-only exploration: volume anomaly',points=False,color_discrete_sequence=['#22a38b','#477ec5'])
    figures.append(fig2)
    fig3=px.scatter(sample,x='return_1m',y='volume_z',color='Class',opacity=.45,title='Training-only exploration: price and volume',color_discrete_sequence=['#22a38b','#477ec5'])
    figures.append(fig3)
    source_table=pd.DataFrame(report['splits']).T
    outcome=pd.read_csv(ARTIFACTS/'event_results.csv')
    ev=report['event_evaluation']
    winner=report['selected_model']; m=report['models'][winner]['test']
    document=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>PumpWatch research report</title>
<style>body{{font:16px/1.65 system-ui,sans-serif;max-width:1100px;margin:45px auto;padding:0 24px;color:#16304a;background:#f5f8fb}}h1{{font-size:44px;letter-spacing:-2px}}h2{{margin-top:42px}}table{{border-collapse:collapse;width:100%;font-size:14px;background:white}}th,td{{padding:10px;border:1px solid #d9e3ed;text-align:left}}.note{{padding:18px;background:#e4eef8;border-left:4px solid #277b9d}}.hero{{color:#197d70;letter-spacing:3px;font-size:13px}}code{{background:#e2eaf2;padding:2px 5px}}</style>
<div class="hero">PUMPWATCH · REPRODUCIBLE RESEARCH</div><h1>Detection of cryptocurrency pump events</h1>
<p>A completed implementation of the synopsis: four classical classifiers, causal OHLCV features, chronological evaluation, and an interactive local dashboard.</p>
<div class="note"><b>{winner}</b> was selected using validation F1. Test precision: <b>{m['precision']:.1%}</b>; recall: <b>{m['recall']:.1%}</b>; F1: <b>{m['f1']:.3f}</b>. Event recall: <b>{ev['detected']}/{ev['events']}</b>. Scores are uncalibrated and do not establish manipulation.</div>
<h2>1. Data audit</h2><p>{audit['matched_onsets']} documented onsets; {report['event_coverage']['complete']} fully usable events; {audit['archives_usable']} of {audit['archives_requested']} requested daily archives; {audit['candle_rows']:,} raw candles; {audit['coins']} pairs. Eight unavailable archives concern YOYOWBTC. Sources cover {html.escape(audit['first_candle'])} through {html.escape(audit['last_candle'])}.</p>
<p>Research onset labels come from the authors' 25-second labeled feature file. Only its symbol, event identifier and positive timestamp are used: published feature values are not model inputs. All onsets match a Binance announcement within five minutes. Daily candle archives are SHA-256 verified against Binance checksums. A source snapshot, hashes, exclusions and full download manifest are included.</p>
<h2>2. Target and timing</h2><p>The minute containing a manual onset and the next two minutes are positive. Other minutes within ±60 minutes of every known onset or announcement are excluded. Negative windows mean no known event, not verified absence of manipulation. Predictions are available at candle close (open timestamp plus one minute). Sixty complete prior candles are required. Missing intervals restart this warm-up.</p>
<p>Full event days and the same pairs seven days earlier supply context and comparison windows. The sample is event-conditioned; it is not representative of the whole market. A three-minute target is a project design choice, not the original paper's exact task.</p>
<h2>3. Chronological experiment</h2>{source_table.to_html(border=0)}
<p>Boundaries follow approximately 60/20/20% of unique event dates. Whole events remain within one partition; feature lookback and target windows are purged at boundaries. All positive training rows and up to 10,000 randomly sampled negative training rows are used. Validation/test retain eligible negatives. Scaling and weights use training data only. Two candidate settings per classifier are compared on a single chronological validation holdout. Validation F1 selects both parameters and alert thresholds. There is no random row cross-validation. The selected model is not refit on validation data before testing.</p>
<h2>4. Results</h2>{frame.round(4).to_html(index=False,border=0)}
<p>Average precision summarizes the stepwise precision–recall curve (not trapezoidal PR integration). Minority-class metrics, independent-event recall, and false alerts are emphasized over accuracy.</p>
<p>{ev['detected']} of {ev['events']} test events were detected within the three-minute target. Median close-time delay among detected events: {ev['median_delay_seconds']} seconds. False alert episodes: {ev['false_alert_episodes']}; exposure: {ev['eligible_negative_coin_days']:.2f} eligible negative coin-days; rate: {ev['false_alert_episodes_per_coin_day']:.3f} episodes per coin-day. Consecutive alert minutes count as one episode. Ambiguous minutes are excluded from both numerator and denominator.</p>
<h2>5. Exploration and model comparison</h2>'''
    for i,fig in enumerate(figures):
        document+=pio.to_html(fig,full_html=False,include_plotlyjs=True if i==0 else False)
    document+=f'''<h2>6. Errors and limitations</h2><p>False-positive and false-negative windows are exported in <code>artifacts/errors.csv</code>; event-level outcomes are below. Missed and late detections remain part of the results. No performance target was imposed after seeing test data.</p>{outcome.to_html(index=False,border=0)}
<p>Limitations: old market regimes; imperfect manual timing; one-minute resolution; missing YOYOW data; no order-book features; a finite number of independent events; no confidence intervals or deployment-wide prevalence estimate; threshold selection on one validation period. No claim of advance prediction is made. A close-time alert can occur after the price move has started. False alerts may include unrecorded events.</p>
<p>The main model uses trade activity where available. An independently fitted OHLCV-only fallback uses the selected algorithm and a separately selected validation threshold; its test metrics are recorded separately. Unseen-coin results are also available in the evaluation JSON. Local explanations replace each feature with its training median one at a time; they are non-additive sensitivity measurements, not causal explanations.</p>
<h2>7. Reproduction and demonstration</h2><p>Run <code>python -m pumpwatch.data</code>, <code>python -m pumpwatch.train</code>, then <code>python -m pumpwatch.reporting</code>. Start the dashboard with <code>python -m streamlit run app.py</code>. The included trained artifacts allow an offline demonstration without retraining. Use Historical explorer, Model comparison, CSV analysis and Historical replay. Setup, tests and commands are in README.md.</p>
<h2>References</h2><p>La Morgia, Mei, Sassi and Stefa (2020), <i>Pump and Dumps in the Bitcoin Era: Real Time Detection of Cryptocurrency Market Manipulations</i>, ICCCN, DOI 10.1109/ICCCN49398.2020.9209660. <a href="https://github.com/SystemsLab-Sapienza/pump-and-dump-dataset">Dataset and source license</a>. <a href="https://github.com/binance/binance-public-data">Binance archive documentation and terms</a>.</p></html>'''
    (ROOT/'Research_Report.html').write_text(document,encoding='utf-8')
    notebook=nbf.v4.new_notebook()
    notebook.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
    cells=[nbf.v4.new_markdown_cell('# PumpWatch exploratory analysis\nReal Binance data and independently published onset labels. Training-only exploration; held-out results are reported separately. Run from this project folder.'),
           nbf.v4.new_code_cell("from pathlib import Path\nimport json\nimport pandas as pd\nimport plotly.express as px\nroot = Path.cwd()\nif not (root / 'artifacts').exists():\n    root = root.parent\nd = pd.read_csv(root / 'artifacts/predictions.csv.gz')\nreport = json.loads((root / 'artifacts/report.json').read_text())\npd.DataFrame(report['splits']).T"),
           nbf.v4.new_markdown_cell('## Class balance and feature distributions\nSynthetic test fixtures never enter this analysis. Normal means no known event. Excluded neighborhoods have target -1.'),
           nbf.v4.new_code_cell("d.groupby(['split','target']).size().rename('windows').to_frame()"),
           nbf.v4.new_code_cell("training = d[(d.split == 'train') & (d.target >= 0)]\ntraining.groupby('target')[['return_1m','volume_z','volatility_15m','trade_z']].median()"),
           nbf.v4.new_code_cell("sample = pd.concat([training.query('target == 1'), training.query('target == 0').sample(3000, random_state=42)])\npx.box(sample, x='target', y='volume_z', color='target', points=False, title='Training volume anomalies')"),
           nbf.v4.new_markdown_cell('## Held-out comparison\nThe model and thresholds were selected before final test evaluation.'),
           nbf.v4.new_code_cell("pd.read_csv(root / 'artifacts/model_comparison.csv')"),
           nbf.v4.new_code_cell("pd.read_csv(root / 'artifacts/event_results.csv')"),
           nbf.v4.new_markdown_cell('## Interpretation\nScores are uncalibrated. Event-conditioned sampling, historical regimes and uncertain labels limit generalization. See Research_Report.html and artifacts/errors.csv for the error analysis.')]
    notebook.cells=cells
    nbf.write(notebook,ROOT/'Exploratory_Analysis.ipynb')
    # Real held-out event example includes sufficient preceding warm-up context.
    chosen=outcome[outcome.detected].iloc[0] if outcome.detected.any() else outcome.iloc[0]
    candles=pd.read_csv(DATA/'candles.csv.gz')
    candles['timestamp']=pd.to_datetime(candles.timestamp,utc=True)
    onset=pd.Timestamp(chosen.onset).floor('min')
    example=candles[(candles.symbol==chosen.pair)&candles.timestamp.between(onset-pd.Timedelta(minutes=120),onset+pd.Timedelta(minutes=30))]
    (ROOT/'examples').mkdir(exist_ok=True)
    example[['timestamp','symbol','open','high','low','close','volume','trades','taker_buy_volume']].to_csv(ROOT/'examples/sample_candles.csv',index=False)
    example[['timestamp','symbol','open','high','low','close','volume']].to_csv(ROOT/'examples/sample_ohlcv_only.csv',index=False)
    print('Report, notebook, coverage audit, error tables, and real demo samples generated.',flush=True)


if __name__=='__main__':
    generate()
