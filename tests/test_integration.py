import json
import unittest
import joblib
import numpy as np
import pandas as pd
from pumpwatch.data import ROOT
from pumpwatch.inference import predict, explain


@unittest.skipUnless((ROOT/'artifacts/best_model.joblib').exists() and (ROOT/'examples/sample_candles.csv').exists(),'Train and generate sample artifacts first')
class TrainedIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle=joblib.load(ROOT/'artifacts/best_model.joblib')
        cls.candles=pd.read_csv(ROOT/'examples/sample_candles.csv')

    def test_replay_prefix_scores_equal_full_history(self):
        full,_=predict(self.candles,self.bundle)
        prefix,_=predict(self.candles.iloc[:100],self.bundle)
        overlap=full[full.timestamp.isin(prefix.timestamp)]
        np.testing.assert_allclose(overlap.risk_score,prefix.risk_score,atol=1e-12)

    def test_saved_reload_reproduces_predictions(self):
        a,_=predict(self.candles,self.bundle)
        b,_=predict(self.candles,joblib.load(ROOT/'artifacts/best_model.joblib'))
        np.testing.assert_array_equal(a.risk_score,b.risk_score)

    def test_ohlcv_fallback_and_explanation(self):
        core=self.candles.drop(columns=['trades','taker_buy_volume'])
        p,q=predict(core,self.bundle)
        self.assertEqual(q['feature_mode'],'OHLCV only')
        self.assertEqual(q['threshold'],self.bundle['core_threshold'])
        self.assertTrue(p.risk_score.between(0,1).all())
        explanation=explain(p.iloc[-1],self.bundle,core=True)
        self.assertEqual(len(explanation),len(self.bundle['core_features']))
        self.assertTrue(np.isfinite(explanation.score_change).all())

    def test_insufficient_history_is_clear(self):
        with self.assertRaisesRegex(ValueError,'61 consecutive'):
            predict(self.candles.iloc[:60],self.bundle)

    def test_report_and_model_selection_agree(self):
        r=json.loads((ROOT/'artifacts/report.json').read_text())
        winner=max(r['models'],key=lambda n:r['models'][n]['validation']['f1'])
        self.assertEqual(winner,self.bundle['name'])
        self.assertEqual(self.bundle['threshold'],r['models'][winner]['validation']['threshold'])
        counts=r['models'][winner]['test']['confusion_matrix']
        self.assertEqual(sum(sum(row) for row in counts),r['splits']['test']['rows'])


if __name__=='__main__':
    unittest.main()
