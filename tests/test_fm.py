import mylibml
import numpy as np
import pandas as pd
import os
from collections import namedtuple
from sklearn.metrics import mean_squared_error
import unittest

class TestFactorizationMachines(unittest.TestCase):
    def setUp(self):
        path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'resources/coat')
        FMsDS = namedtuple('FMsDS', ('X', 'y'))

        RAW_TRAIN = pd.read_csv(
            os.path.join(path, 'train.ascii'), header=None, delimiter=' '
        )

        TRAIN = RAW_TRAIN.reset_index().pipe(
            lambda df: pd.melt(df, id_vars=['index'], value_vars=list(range(300)))
        ).rename(columns={'index': 'user_id', 'variable': 'item_id', 'value': 'rating'}).pipe(
            lambda df: df[df.rating > 0]
        ).reset_index(drop=True)

        RAW_TEST = pd.read_csv(
            os.path.join(path, 'test.ascii'), header=None, delimiter=' '
        )

        TEST = RAW_TEST.reset_index().pipe(
            lambda df: pd.melt(df, id_vars=['index'], value_vars=list(range(300)))
        ).rename(columns={'index': 'user_id', 'variable': 'item_id', 'value': 'rating'}).pipe(
            lambda df: df[df.rating > 0]
        ).reset_index(drop=True)

        RAW_ITEM = pd.read_csv(
            os.path.join(path, 'item_features.ascii'), header=None, delimiter=' '
        )

        ITEM = RAW_ITEM.add_prefix('item_f_').reset_index().rename(
            columns={'index': 'item_id'}
        )

        RAW_USER = pd.read_csv(
            os.path.join(path, 'user_features.ascii'), header=None, delimiter=' '
        )

        USER = RAW_USER.add_prefix('user_f_').reset_index().rename(
            columns={'index': 'user_id'}
        )

        TRAIN_MERGED_FEATURES = TRAIN.pipe(
            lambda df: pd.merge(df, USER, on='user_id')
        ).pipe(
            lambda df: pd.merge(df, ITEM, on='item_id')
        )

        TEST_MERGED_FEATURES = TEST.pipe(
            lambda df: pd.merge(df, USER, on='user_id')
        ).pipe(
            lambda df: pd.merge(df, ITEM, on='item_id')
        )

        self.FMs_TRAIN = TRAIN_MERGED_FEATURES.pipe(
            lambda df: pd.get_dummies(df, columns=['user_id', 'item_id'])
        ).pipe(
            lambda df: FMsDS(
                df.drop('rating', axis=1),
                df['rating']
            )
        )

        self.FMs_TEST = TEST_MERGED_FEATURES.pipe(
            lambda df: pd.get_dummies(df, columns=['user_id', 'item_id'])
        ).pipe(
            lambda df: FMsDS(
                df.drop('rating', axis=1)[self.FMs_TRAIN.X.columns],
                df['rating']
            )
        )

    def test_fit(self):
        FMs = mylibml.fm.FactorizationMachines(K=5)
        FMs.fit(self.FMs_TRAIN.X.values, self.FMs_TRAIN.y.values)
        y_pred = FMs.predict(self.FMs_TEST.X.values)
        error = mean_squared_error(y_pred, self.FMs_TEST.y.values)
        print('FMs error =>', end='')
        print(error)

class TestPropensityFactorizationMachines(TestFactorizationMachines):
    def test_fit(self):
        PFMs = mylibml.fm.PropensityFactorizationMachines(K=5)
        PFMs.fit(self.FMs_TRAIN.X.assign(propensity=1).values, self.FMs_TRAIN.y.values)
        y_pred = PFMs.predict(self.FMs_TEST.X.values)
        error = mean_squared_error(y_pred, self.FMs_TEST.y.values)
        print('PFMs error =>', end='')
        print(error)

if __name__ == '__main__':
    unittest.main()
