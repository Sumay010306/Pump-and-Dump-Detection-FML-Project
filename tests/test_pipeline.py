"""Synthetic fixtures test mechanics only; model evaluation uses real data."""
import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from pumpwatch.data import validate_candles
from pumpwatch.features import FEATURES, engineer, label_windows
from pumpwatch.train import split_by_time, choose_threshold


def fixture(n=180):
    close=100+np.arange(n)*.01
    return pd.DataFrame({'timestamp':pd.date_range('2020-01-01',periods=n,freq='min',tz='UTC'),
        'open':close,'high':close+.02,'low':close-.02,'close':close,
        'volume':np.arange(n)%11+1.,'trades':np.arange(n)%5+1,
        'taker_buy_volume':(np.arange(n)%11+1.)*.6,'symbol':'TESTBTC'})


class PipelineTests(unittest.TestCase):
    def test_future_cannot_change_past_features(self):
        d=fixture();a=engineer(d)
        changed=d.copy(); changed.loc[100:,'volume']=999999
        changed.loc[100:,'close']*=4
        b=engineer(changed)
        assert_frame_equal(a.loc[:99,FEATURES],b.loc[:99,FEATURES])

    def test_prefix_features_equal_batch_features(self):
        d=fixture()
        assert_frame_equal(engineer(d).loc[:99,FEATURES],engineer(d.iloc[:100])[FEATURES])

    def test_baseline_excludes_current_candle(self):
        d=fixture(61);d['volume']=1.;d.loc[60,'volume']=10
        f=engineer(d)
        self.assertEqual(f.iloc[60].volume_ratio,10)
        self.assertFalse(f.iloc[59].ready)
        self.assertTrue(f.iloc[60].ready)

    def test_gap_restarts_warmup(self):
        d=fixture().drop(index=80)
        f=engineer(d)
        self.assertFalse(f.iloc[80].ready)
        self.assertFalse(f.iloc[139].ready)
        self.assertTrue(f.iloc[140].ready)

    def test_epoch_units_agree(self):
        d=fixture()
        for divisor in [10**9,10**6,10**3]:
            f=d.copy()
            f['timestamp']=d.timestamp.astype('datetime64[ns, UTC]').astype('int64')//divisor
            parsed,_=validate_candles(f)
            self.assertEqual(parsed.timestamp.iloc[0],d.timestamp.iloc[0])

    def test_duplicate_policy(self):
        d=fixture()
        clean,quality=validate_candles(pd.concat([d,d.iloc[:1]]))
        self.assertEqual(quality['duplicates_removed'],1)
        conflicting=d.iloc[:1].copy();conflicting['volume']=222
        with self.assertRaisesRegex(ValueError,'Conflicting'):
            validate_candles(pd.concat([d,conflicting]))

    def test_bad_prices_rejected(self):
        d=fixture();d.loc[0,'high']=1
        with self.assertRaisesRegex(ValueError,'high/low'):
            validate_candles(d)

    def test_wrong_interval_rejected(self):
        with self.assertRaisesRegex(ValueError,'interval'):
            validate_candles(fixture().iloc[::5])

    def test_mixed_pairs_rejected(self):
        d=fixture();d.loc[0,'symbol']='OTHER'
        with self.assertRaisesRegex(ValueError,'one trading pair'):
            validate_candles(d)

    def test_label_duration_and_exclusion(self):
        d=engineer(fixture())
        onset=pd.Timestamp('2020-01-01 01:30:25',tz='UTC')
        e=pd.DataFrame([{'onset':onset,'pair':'TESTBTC','event_id':'e'}])
        a=pd.DataFrame([{'announced':onset,'symbol':'TEST'}])
        f=label_windows(d,e,a)
        self.assertEqual(int((f.target==1).sum()),3)
        self.assertEqual(f.iloc[89].target,-1)
        self.assertEqual(f.iloc[93].target,-1)
        self.assertEqual(f.iloc[0].target,0)
        self.assertEqual(f.iloc[90].prediction_time,pd.Timestamp('2020-01-01 01:31',tz='UTC'))

    def test_chronological_partition_and_purge(self):
        events=pd.DataFrame({'onset':pd.date_range('2020-01-01',periods=20,freq='D',tz='UTC')})
        d=pd.DataFrame({'timestamp':pd.date_range('2020-01-01',periods=20*1440,freq='min',tz='UTC'),'event_id':''})
        f,b=split_by_time(d,events)
        self.assertLess(f[f.split=='train'].timestamp.max(),f[f.split=='validation'].timestamp.min())
        self.assertLess(f[f.split=='validation'].timestamp.max(),f[f.split=='test'].timestamp.min())
        cutoff=pd.Timestamp(b['validation_start'])
        self.assertTrue((f[f.timestamp.between(cutoff,cutoff+pd.Timedelta(minutes=60))].split=='purged').all())

    def test_threshold_selection(self):
        self.assertEqual(choose_threshold([0,0,1,1],[.1,.2,.8,.9]),.8)


if __name__=='__main__':
    unittest.main()
