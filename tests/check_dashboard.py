"""Optional UI smoke check: python tests/check_dashboard.py"""
from pathlib import Path
import sys
from streamlit.testing.v1 import AppTest

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
app=AppTest.from_file(str(root/'app.py'),default_timeout=60).run()
assert not app.exception, [e.message for e in app.exception]
for screen in ['Historical explorer','CSV analysis','Model comparison','Historical replay','Data & methodology']:
    app.sidebar.radio[0].set_value(screen).run()
    assert not app.exception,[e.message for e in app.exception]
    print(screen, 'passed')
app.sidebar.radio[0].set_value('Historical replay').run()
app.slider[0].set_value(130).run()
assert not app.exception,[e.message for e in app.exception]
print('Replay slider interaction passed')
