"""Editable dashboard wording; changing these strings does not retrain any model.

Keep dictionary keys (such as 'overview' and 'title') unchanged. The application
uses those stable keys for navigation, while displaying the values to readers.
"""

# Browser tab and sidebar identity.
SITE = {
    'browser_title': 'PumpWatch | Market research',
    'sidebar_title': '◈ PumpWatch',
    'sidebar_caption': 'CRYPTOCURRENCY MARKET RESEARCH',
    'navigation_label': 'Workspace',
    'sidebar_context': 'Historical research · one-minute candles · UTC',
    'sidebar_scope': 'Historical dataset evaluation · no live market feed.',
}

# Change labels/titles here freely; keep page keys unchanged for navigation.
PAGES = {
    'overview': {
        'label': 'Overview',
        'title': 'Spot the surge. Understand the signal.',
        'method': 'Supervised binary classification learns patterns from historical windows labeled as pump activity or no known event. Four classifiers are trained on earlier data, selected using validation F1, and evaluated on a later test period.',
        'meaning': 'Test precision is the share of alert windows matching labeled pump activity. Event recall is the share of independent test events detected at least once in their three-minute target window.',
    },
    'explorer': {
        'label': 'Historical explorer',
        'title': 'Explore a documented event',
        'method': 'The selected model uses price returns, volume anomalies, volatility and trade activity from completed candles. The chart displays its saved scores for the selected historical event.',
        'meaning': 'Diamonds mark scores above the validation-selected threshold; the dotted line marks the documented onset. The feature chart measures how replacing one observed value with its training median changes the score. Training examples are descriptive; held-out test events provide the final evaluation.',
    },
    'comparison': {
        'label': 'Model comparison',
        'title': 'Evidence before confidence',
        'method': 'Logistic Regression, Random Forest, RBF SVM and Gradient Boosting are compared using the same chronological partitions. Model settings and alert thresholds are selected on validation data before the test results are examined.',
        'meaning': 'Precision measures alert correctness, recall measures captured pump windows, and F1 balances the two. Average precision summarizes the precision–recall curve. Higher values are better, but false alerts and event-level detection also matter.',
    },
    'replay': {
        'label': 'Historical replay',
        'title': 'Replay the market',
        'method': 'The already-trained model is applied to a growing prefix of the historical event. Features are recomputed using only the candles revealed so far; the model is not retrained during replay.',
        'meaning': 'An alert appears when the current score crosses the validation-selected threshold. Scores become available at candle close. This demonstrates historical detection timing, rather than predicting an event before it starts.',
    },
    'methodology': {
        'label': 'Data & methodology',
        'title': 'Data & methodology',
        'method': 'A chronological train–validation–test split keeps learning, model selection and final evaluation separate. Earlier data trains the models, validation chooses settings, and later test data measures generalization.',
        'meaning': 'The table shows the actual windows, pump windows and independent events in each partition. Boundaries follow approximately 60/20/20% of event dates; window counts differ because coverage varies and unusable windows are excluded.',
    },
}

# Frequently edited overview copy and section headings.
TEXT = {
    'overview_eyebrow': 'PATTERNS, EVIDENCE, EXPLANATIONS',
    'overview_subtitle': 'Explore documented pump events, compare four machine-learning models, and inspect the evidence behind every alert.',
    'overview_comparison': 'Four models. One held-out comparison.',
    'overview_navigation': 'Start exploring',
    'overview_links': '**Historical explorer** — inspect real price spikes and model alerts.\n\n**Historical replay** — reveal candles sequentially, with no future data supplied to the detector.\n\n**Model comparison** — review errors, precision–recall curves, and baselines.',
    'score_notice': 'Scores identify similarity to labeled pump activity. They are not calibrated probabilities. Results describe historical event/comparison days, not all cryptocurrency trading.',
    'confusion_title': 'Confusion matrix',
    'pr_title': 'Test precision–recall curves',
    'event_title': 'Event detection and false alerts · selected model, test period',
    'baseline_title': 'Reference methods · test period',
    'importance_title': 'Global feature importance',
}

# Explanations describe the actual algorithms in train.py, not causal claims.
MODEL_EXPLANATIONS = {
    'Logistic Regression': 'Logistic Regression combines scaled features with learned weights to form a linear classification boundary.',
    'Random Forest': 'Random Forest averages many decision trees, allowing it to capture nonlinear combinations of price and trading activity.',
    'SVM': 'The RBF support-vector machine uses a nonlinear boundary to separate pump and normal feature patterns. Its margin is mapped to a score; that score is not a calibrated probability.',
    'Gradient Boosting': 'Gradient Boosting builds decision trees sequentially, with later trees correcting errors made by the earlier ensemble.',
}
