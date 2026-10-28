"""Exercise every historical dashboard page and ensure external uploads stay absent."""
from pathlib import Path
import sys
from streamlit.testing.v1 import AppTest

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from pumpwatch.dashboard_text import PAGES

# Test the fully built project; a source-only clone first needs its artifacts.
app=AppTest.from_file(str(root/'app.py'),default_timeout=60).run()
assert not app.exception, [e.message for e in app.exception]
assert len(app.sidebar.radio[0].options)==5
for screen in PAGES:
    # Stable IDs keep navigation working when the display labels are edited.
    app.sidebar.radio[0].set_value(screen).run()
    assert not app.exception,[e.message for e in app.exception]
    assert not app.get('file_uploader'), 'An upload control should never be exposed.'
    captions=[c.value for c in app.caption]
    assert any('**Method:**' in c for c in captions)
    assert any('**What this means:**' in c for c in captions)
    print(PAGES[screen]['label'], 'passed: explanation present, no upload controls')
app.sidebar.radio[0].set_value('replay').run()
app.slider[0].set_value(130).run()
assert not app.exception,[e.message for e in app.exception]
print('Replay slider interaction passed')

# Verify that switching partition and algorithm still updates the comparison page.
app.sidebar.radio[0].set_value('comparison').run()
next(widget for widget in app.radio if widget.label=='Evaluation period').set_value('validation').run()
assert not app.exception,[e.message for e in app.exception]
app.selectbox[0].set_value('SVM').run()
assert not app.exception,[e.message for e in app.exception]
print('Validation selector and model-specific explanation passed')
