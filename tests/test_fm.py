import mylibml
import numpy as np
import pandas as pd
import os
from collections import namedtuple
from sklearn.metrics import mean_squared_error
import unittest

class TestFactorizationMachines(unittest.TestCase):
    def test_fit(self):
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

        FMs_TRAIN = TRAIN_MERGED_FEATURES.pipe(
            lambda df: pd.get_dummies(df, columns=['user_id', 'item_id'])
        ).pipe(
            lambda df: FMsDS(
                df.drop('rating', axis=1),
                df['rating']
            )
        )

        FMs_TEST = TEST_MERGED_FEATURES.pipe(
            lambda df: pd.get_dummies(df, columns=['user_id', 'item_id'])
        ).pipe(
            lambda df: FMsDS(
                df.drop('rating', axis=1)[FMs_TRAIN.X.columns],
                df['rating']
            )
        )

        FMs = mylibml.fm.FactorizationMachines(K=5, λ=1, LOOP=3)
        FMs.fit(FMs_TRAIN.X.values, FMs_TRAIN.y.values)
        y_pred = FMs.predict(FMs_TEST.X.values)
        error = mean_squared_error(y_pred, FMs_TEST.y.values)
        print(error)
        print(FMs.score(FMs_TEST.X.values, FMs_TEST.y.values))

if __name__ == '__main__':
    unittest.main()

