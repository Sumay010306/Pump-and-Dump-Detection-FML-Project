# Edit dashboard headings and explanations

Use this Git-ready project as the source for future edits. There is also a separate runnable copy in the sibling `pumpwatch` folder. Both were updated together, but future edits to one copy do not automatically update the other. Run the copy you edited, or copy the changed files across.

## Most wording: `pumpwatch/dashboard_text.py`

This file separates frequently edited text from the ML calculations.

| What to change | Dictionary entry |
|---|---|
| Browser tab title | `SITE['browser_title']` |
| Sidebar project name | `SITE['sidebar_title']` |
| Sidebar subtitle | `SITE['sidebar_caption']` |
| Sidebar navigation heading | `SITE['navigation_label']` |
| Name of a page in navigation | `PAGES['overview']['label']`, or another page ID |
| Main heading on a page | `PAGES['overview']['title']`, or another page ID |
| Explanation of the ML method | `PAGES['overview']['method']`, or another page ID |
| Explanation of the result | `PAGES['overview']['meaning']`, or another page ID |
| Overview introduction | `TEXT['overview_subtitle']` |
| Small uppercase overview text | `TEXT['overview_eyebrow']` |
| Overview chart heading | `TEXT['overview_comparison']` |
| Overview navigation descriptions | `TEXT['overview_links']` |
| Confusion matrix / PR headings | `TEXT['confusion_title']`, `TEXT['pr_title']` |
| Event / baseline / importance headings | `TEXT['event_title']`, `TEXT['baseline_title']`, `TEXT['importance_title']` |
| Algorithm explanation | The matching value in `MODEL_EXPLANATIONS` |

Page IDs are `overview`, `explorer`, `comparison`, `replay`, and `methodology`.

For example, inside the existing `overview` entry, replace:

```python
'title': 'Spot the surge. Understand the signal.',
```

with:

```python
'title': 'Cryptocurrency Pump-and-Dump Detection',
```

To rename the sidebar item, change its value:

```python
'label': 'Project Overview',
```

Keep keys such as `overview`, `label` and `title` unchanged. Navigation uses the keys internally, so editing the displayed label does not break it. If you rename a page, also adjust its manually written mention in `TEXT['overview_links']`.

Preserve quotes, commas and brackets. For text containing an apostrophe, double quotes are convenient:

```python
'meaning': "This page shows the model's results on later historical data.",
```

## Other labels and charts: `app.py`

Search for the exact wording you see on screen. Typical display calls include `st.metric`, `st.selectbox`, `st.caption`, and `st.expander`. Change their quoted labels while leaving value expressions intact. For example, change `st.metric('Test precision', ...)` to `st.metric('Precision on test data', ...)` without changing its calculation.

- `chart()` controls chart legend and axis titles.
- `detail()` controls prediction cards and feature-explanation captions.
- The page blocks are marked `OVERVIEW`, `HISTORICAL VIEWS`, `MODEL COMPARISON`, and `METHODOLOGY`.
- `page_explanation()` controls the prefixes “Method” and “What this means”.
- Internal data keys such as `risk_score`, `target`, `split`, `precision`, and model names in `report['models']` must stay unchanged for wording-only edits.

## Appearance

Edit `.streamlit/config.toml` for theme colors. The CSS block near the top of `app.py` controls card backgrounds, spacing and heading styles. Restart Streamlit after changing configuration if the theme does not refresh.

## See and verify changes

From the edited project directory, run:

```powershell
python -m streamlit run app.py
```

If Streamlit is already open, save the file and use **Rerun** or refresh the browser. A source-only checkout needs the existing data/model artifacts restored or built first. You do not need to train again for comments, titles, captions, colors or layout changes.

To check all five pages after editing:

```powershell
python tests/check_dashboard.py
```

The dashboard has five pages and no file-upload controls. Its historical train/validation/test results, trained models and thresholds were preserved during this change.
