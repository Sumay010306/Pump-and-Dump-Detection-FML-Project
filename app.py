"""Local Streamlit research dashboard. Run: python -m streamlit run app.py"""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from pumpwatch.data import ROOT, DATA
from pumpwatch.inference import predict, explain

st.set_page_config(page_title="PumpWatch | Market research", page_icon="◈", layout="wide")
st.markdown("""<style>
.stApp {background:#09121d;color:#e6edf5} h1,h2,h3 {letter-spacing:-.035em}
[data-testid="stSidebar"] {background:#101d2c}
[data-testid="stMetric"] {background:#112236;border:1px solid #24394e;border-radius:12px;padding:16px}
.eyebrow {font-size:12px;letter-spacing:3px;color:#50d6b1;margin-top:8px}
.subtle {color:#9fb2c8;font-size:16px;max-width:850px}
.stButton>button {border-radius:8px}
</style>""", unsafe_allow_html=True)

ART = ROOT / "artifacts"
required_artifacts = [ART/'report.json', ART/'best_model.joblib', ART/'predictions.csv.gz',
                      DATA/'candles.csv.gz', DATA/'events.csv', DATA/'audit.json']
if not all(path.exists() for path in required_artifacts):
    st.title("PumpWatch")
    st.info("This source checkout needs the datasets and trained models. Follow the setup guide to build them, or restore them from the complete project package.")
    st.code("python -m pumpwatch.data\npython -m pumpwatch.train\npython -m pumpwatch.reporting")
    st.stop()


@st.cache_resource
def load_bundle():
    # Only load the trusted model shipped with this project; never accept model uploads.
    return joblib.load(ART / 'best_model.joblib')


@st.cache_data
def load_reports():
    return json.loads((ART / 'report.json').read_text()), json.loads((DATA / 'audit.json').read_text())


@st.cache_data
def load_predictions():
    d = pd.read_csv(ART / 'predictions.csv.gz')
    for c in ['timestamp', 'prediction_time']:
        d[c] = pd.to_datetime(d[c], utc=True)
    return d


@st.cache_data
def load_candles():
    d = pd.read_csv(DATA / 'candles.csv.gz')
    d['timestamp'] = pd.to_datetime(d.timestamp, utc=True)
    return d


report, audit = load_reports()
bundle = load_bundle()
best = report['models'][report['selected_model']]['test']


def chart(d, threshold, onset=None):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=.04, row_heights=[.48,.24,.28])
    fig.add_trace(go.Candlestick(x=d.timestamp, open=d.open, high=d.high, low=d.low, close=d.close,
                                name='Price · BTC', increasing_line_color='#50d6b1', decreasing_line_color='#ff7a88'), row=1,col=1)
    fig.add_trace(go.Bar(x=d.timestamp,y=d.volume,name='Base-asset volume',marker_color='#446e98'),row=2,col=1)
    if 'risk_score' in d:
        fig.add_trace(go.Scatter(x=d.timestamp,y=d.risk_score,name='Risk score',line=dict(color='#f8c86a',width=2)),row=3,col=1)
        alerts = d[d.alert.fillna(False)]
        fig.add_trace(go.Scatter(x=alerts.timestamp,y=alerts.close,mode='markers',name='Alert',marker=dict(color='#ff7a88',size=9,symbol='diamond')),row=1,col=1)
    fig.add_hline(y=threshold,line_dash='dash',line_color='#ff7a88',row=3,col=1)
    if onset is not None:
        for row in [1,2,3]:
            fig.add_vline(x=pd.Timestamp(onset).timestamp()*1000,line_dash='dot',line_color='#50d6b1',row=row,col=1)
    fig.update_layout(height=630,template='plotly_dark',paper_bgcolor='#09121d',plot_bgcolor='#101d2c',
                      margin=dict(l=10,r=10,t=25,b=10),legend=dict(orientation='h',y=1.07),xaxis_rangeslider_visible=False)
    fig.update_yaxes(range=[0,1.02],row=3,col=1)
    fig.update_xaxes(title_text='Candle open time · UTC (score available one minute later)',row=3,col=1)
    return fig


def detail(row, core=False):
    threshold = bundle['core_threshold'] if core else bundle['threshold']
    a,b,c=st.columns(3)
    a.metric('Risk score',f"{row.risk_score:.6f}")
    b.metric('Decision','Pump alert' if row.risk_score >= threshold else 'No alert')
    c.metric('Alert threshold',f'{threshold:.6f}')
    st.caption(f"Available at {row.prediction_time} · Uncalibrated score, not a probability or proof of manipulation.")
    explanation=explain(row,bundle,core)
    top=explanation.head(8).sort_values('score_change')
    fig=px.bar(top,x='score_change',y='feature',orientation='h',color='score_change',
               color_continuous_scale=['#6c9de9','#50d6b1'],template='plotly_dark')
    fig.update_layout(height=310,paper_bgcolor='#09121d',plot_bgcolor='#101d2c',coloraxis_showscale=False,
                      xaxis_title='Score change when observed value replaces training median',yaxis_title='')
    st.plotly_chart(fig,width='stretch')
    st.caption('One-feature-at-a-time sensitivity: positive values support the alert. Values do not add up to the score; correlated features can produce unrealistic substitutions. This is not causal attribution or SHAP.')
    with st.expander('Inspect feature values and reference values'):
        st.dataframe(explanation,hide_index=True,width='stretch')


st.sidebar.markdown('## ◈ PumpWatch')
st.sidebar.caption('CRYPTOCURRENCY MARKET RESEARCH')
page=st.sidebar.radio('Workspace',['Overview','Historical explorer','CSV analysis','Model comparison','Historical replay','Data & methodology'])
st.sidebar.divider()
st.sidebar.markdown(f"**Selected model**  \n{report['selected_model']}")
st.sidebar.caption('Historical research · one-minute candles · UTC')
st.sidebar.caption('No trading actions or live market feed.')

if page == 'Overview':
    st.markdown('<div class="eyebrow">PATTERNS, EVIDENCE, EXPLANATIONS</div>',unsafe_allow_html=True)
    st.title('Spot the surge. Understand the signal.')
    st.markdown('<p class="subtle">Explore documented pump events, compare four machine-learning models, and inspect the evidence behind every alert.</p>',unsafe_allow_html=True)
    a,b,c,d=st.columns(4)
    a.metric('Documented onsets',audit['matched_onsets'])
    b.metric('Trading pairs',audit['coins'])
    c.metric('Test precision',f"{best['precision']:.1%}")
    d.metric('Test event recall',f"{report['event_evaluation']['recall']:.1%}")
    st.info('Scores identify similarity to labeled pump activity. They are not calibrated probabilities. Results describe historical event/comparison days, not all cryptocurrency trading.')
    left,right=st.columns([1.6,1])
    with left:
        st.subheader('Four models. One held-out comparison.')
        rows=[{'Model':n,'F1':m['test']['f1'],'Precision':m['test']['precision'],'Recall':m['test']['recall']} for n,m in report['models'].items()]
        fig=px.bar(pd.DataFrame(rows),x='Model',y=['F1','Precision','Recall'],barmode='group',template='plotly_dark',color_discrete_sequence=['#50d6b1','#6c9de9','#f8c86a'])
        fig.update_layout(paper_bgcolor='#09121d',plot_bgcolor='#101d2c',height=360,yaxis_range=[0,1])
        st.plotly_chart(fig,width='stretch')
    with right:
        st.subheader('Start exploring')
        st.markdown('**Historical explorer** — inspect real price spikes and model alerts.\n\n**CSV analysis** — score your own one-minute data.\n\n**Historical replay** — reveal candles sequentially, with no future data supplied to the detector.\n\n**Model comparison** — review errors, precision–recall curves, and baselines.')
    st.caption(f"Dataset: {audit['first_candle']} to {audit['last_candle']} · Model selected on validation F1, then evaluated on the untouched test period.")

elif page in ['Historical explorer','Historical replay']:
    replay = page == 'Historical replay'
    st.title('Replay the market' if replay else 'Explore a documented event')
    events=pd.read_csv(DATA/'events.csv')
    events['onset']=pd.to_datetime(events.onset,utc=True)
    outcomes=pd.read_csv(ART/'event_results.csv')
    subset=st.selectbox('Event collection',['Held-out test events','All documented events'])
    if subset.startswith('Held-out'):
        events=events[events.event_id.isin(outcomes.event_id)]
    events=events.sort_values('onset',ascending=False)
    event_id=st.selectbox('Event',events.event_id.tolist(),format_func=lambda x: events.set_index('event_id').loc[x,'pair']+' · '+str(events.set_index('event_id').loc[x,'onset']))
    e=events.set_index('event_id').loc[event_id]
    candles=load_candles()
    raw=candles[(candles.symbol==e.pair)&candles.timestamp.between(e.onset.floor('min')-pd.Timedelta(minutes=120),e.onset.floor('min')+pd.Timedelta(minutes=60))].copy()
    if raw.empty:
        st.warning('Raw data for this event could not be downloaded. See the data audit.')
        st.stop()
    if replay:
        st.caption('Historical replay. Each step runs the shared prediction pipeline on the visible prefix only. The known event marker is hidden until its time has passed.')
        slider_key='replay_'+event_id
        position=st.slider('Completed candles revealed',61,len(raw),min(105,len(raw)),key=slider_key)
        visible=raw.iloc[:position]
        try:
            predicted,quality=predict(visible,bundle)
        except ValueError as exc:
            st.warning(str(exc));st.stop()
        selected=predicted.iloc[-1]
        plot=visible.merge(predicted[['timestamp','risk_score','alert']],on='timestamp',how='left')
        marker=e.onset if selected.prediction_time >= e.onset else None
        st.plotly_chart(chart(plot,bundle['threshold'],marker),width='stretch')
        detail(selected)
    else:
        scored=load_predictions()
        scored=scored[(scored.symbol==e.pair)&scored.timestamp.isin(raw.timestamp)]
        plot=raw.merge(scored[['timestamp','risk_score','alert']],on='timestamp',how='left')
        st.caption(f"Source onset: {e.onset} · Green dotted marker · Coin/BTC pair · Prediction split: {', '.join(scored.split.unique())}")
        st.plotly_chart(chart(plot,bundle['threshold'],e.onset),width='stretch')
        if len(scored):
            times=scored.timestamp.tolist()
            default=int(np.argmin(np.abs(scored.timestamp-pd.Timestamp(e.onset))))
            moment=st.select_slider('Inspect a candle',options=times,value=times[default],format_func=lambda t:t.strftime('%H:%M UTC'))
            detail(scored[scored.timestamp==moment].iloc[0])

elif page == 'CSV analysis':
    st.title('Analyze your candles')
    st.write('Upload at least 61 consecutive, completed one-minute candles for a single trading pair.')
    st.caption('Required: timestamp, open, high, low, close, volume. Optional: trades, taker_buy_volume. Timestamps are candle open times in UTC or epoch seconds/ms/us.')
    example=ROOT/'examples'/'sample_candles.csv'
    if example.exists():
        st.download_button('Download real-data example CSV',example.read_bytes(),'sample_candles.csv','text/csv')
    uploaded=st.file_uploader('Choose a candle CSV',type=['csv'])
    if uploaded is not None:
        try:
            frame=pd.read_csv(uploaded)
            result,quality=predict(frame,bundle)
            st.success(f"Scored {len(result):,} candles · {quality['feature_mode']} · {quality['gaps']} gaps detected")
            if quality['feature_mode']=='OHLCV only':
                st.info('Using the separately trained OHLCV-only fallback model and its own validation-selected threshold.')
            plot=result.tail(500)
            st.plotly_chart(chart(plot,quality['threshold']),width='stretch')
            times=result.timestamp.tolist()
            selected=st.select_slider('Inspect a prediction',options=times,value=times[-1])
            detail(result[result.timestamp==selected].iloc[0],core=quality['feature_mode']=='OHLCV only')
            st.download_button('Download predictions',result[['prediction_time','symbol','risk_score','alert']].to_csv(index=False),'pumpwatch_predictions.csv','text/csv')
        except (ValueError,KeyError,pd.errors.ParserError) as exc:
            st.error(str(exc))

elif page == 'Model comparison':
    st.title('Evidence before confidence')
    st.caption(report['selection'])
    split=st.radio('Evaluation period',['test','validation'],horizontal=True)
    rows=[]
    for name,r in report['models'].items():
        m=r[split]
        rows.append({'Model':name,**{k:m[k] for k in ['precision','recall','f1','average_precision','roc_auc','threshold']}})
    st.dataframe(pd.DataFrame(rows).set_index('Model').style.format('{:.4f}'),width='stretch')
    model=st.selectbox('Inspect model',list(report['models']))
    left,right=st.columns(2)
    with left:
        st.subheader('Confusion matrix')
        cm=report['models'][model][split]['confusion_matrix']
        fig=px.imshow(cm,x=['Predicted normal','Predicted pump'],y=['Labeled normal','Labeled pump'],text_auto=True,color_continuous_scale='Teal',template='plotly_dark')
        fig.update_layout(paper_bgcolor='#09121d',height=340)
        st.plotly_chart(fig,width='stretch')
    with right:
        st.subheader('Test precision–recall curves')
        fig=go.Figure()
        for n,r in report['models'].items():
            fig.add_trace(go.Scatter(x=r['pr_curve']['recall'],y=r['pr_curve']['precision'],name=n))
        fig.update_layout(template='plotly_dark',paper_bgcolor='#09121d',plot_bgcolor='#101d2c',height=340,xaxis_title='Recall',yaxis_title='Precision',legend=dict(orientation='h',y=-.3))
        st.plotly_chart(fig,width='stretch')
    st.subheader('Event detection and false alerts · selected model, test period')
    ev=report['event_evaluation']
    a,b,c,d=st.columns(4)
    a.metric('Detected events',f"{ev['detected']} / {ev['events']}")
    b.metric('Median detection delay',f"{ev['median_delay_seconds']:.0f} s" if ev['median_delay_seconds'] is not None else 'N/A')
    c.metric('False alert episodes',ev['false_alert_episodes'])
    d.metric('False episodes / coin-day',f"{ev['false_alert_episodes_per_coin_day']:.2f}")
    st.caption(ev['scope']+' Consecutive alert minutes count as one episode; ambiguous periods are excluded.')
    st.dataframe(pd.read_csv(ART/'event_results.csv'),hide_index=True,width='stretch')
    st.subheader('Reference methods · test period')
    st.dataframe(pd.DataFrame({n:{k:m[k] for k in ['precision','recall','f1','average_precision']} for n,m in report['baselines'].items()}).T,width='stretch')
    st.caption('Average precision for the binary rule has only two score levels. The always-normal ranking baseline equals the positive prevalence.')
    st.subheader('Global feature importance')
    importance=pd.read_csv(ART/'importance.csv').head(12).sort_values('importance')
    fig=px.bar(importance,x='importance',y='feature',orientation='h',error_x='std',template='plotly_dark',color_discrete_sequence=['#50d6b1'])
    fig.update_layout(paper_bgcolor='#09121d',plot_bgcolor='#101d2c',xaxis_title='Validation average-precision decrease after permutation',yaxis_title='')
    st.plotly_chart(fig,width='stretch')
    with st.expander('OHLCV-only fallback and unseen-coin results'):
        st.json({'OHLCV-only test':report['ohlcv_only_test'],'Unseen coins':report['unseen_coins']})

else:
    st.title('Data & methodology')
    st.write('Research labels are independently published manual onset annotations. Raw one-minute candles come from the corresponding Binance BTC pairs.')
    st.dataframe(pd.DataFrame(report['splits']).T,width='stretch')
    st.write('**Labels:** onset minute and next two minutes are positive. Other minutes within ±60 minutes of any known onset or announcement are excluded from binary evaluation.')
    st.write('**Timing:** candle timestamps are open times; predictions become available at the next minute. Gaps restart the 60-minute feature warm-up.')
    st.write('**Selection:** two settings per model; chronological validation selects settings and F1 thresholds. The test period remains untouched during selection. Training negatives are sampled; validation and test negatives are retained.')
    st.write('**Limits:** historical data from 2018–2021, uncertain event labels, event-conditioned sampling, uncalibrated scores, one-minute latency, and unverified negatives. Contemporary or market-wide performance is not established.')
    st.markdown('[Research dataset and paper](https://github.com/SystemsLab-Sapienza/pump-and-dump-dataset) · [Binance archive documentation](https://github.com/binance/binance-public-data)')
    with st.expander('Full acquisition audit'):
        st.json(audit)
    manifest=DATA/'download_manifest.csv'
    st.download_button('Download source and checksum manifest',manifest.read_bytes(),'download_manifest.csv','text/csv')
    st.download_button('Download model evaluation', (ART/'report.json').read_bytes(),'report.json','application/json')
