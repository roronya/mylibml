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

        LRPropensityDS = namedtuple('LRPropensityDS', ('X', 'y'))

        positive = TRAIN_MERGED_FEATURES.assign(
            observed=1,
        ).drop(['user_id', 'item_id', 'rating'], axis=1)

        negative = pd.merge(
            USER[['user_id']].assign(key=0),
            ITEM[['item_id']].assign(key=0),
            on='key'
        ).drop('key', axis=1).pipe(
            lambda df: pd.merge(
                df,
                TRAIN[['user_id', 'item_id']].assign(observed=1),
                how='left',
                on=['user_id', 'item_id']
            ).fillna(0)
        ).pipe(
            lambda df: df[df.observed == 0]
        ).pipe(
            lambda df: pd.merge(df, USER, on='user_id')
        ).pipe(
            lambda df: pd.merge(df, ITEM, on='item_id')
        ).drop(['user_id', 'item_id'], axis=1)[positive.columns]

        LR_PROPENSITY_TRAIN = pd.concat([positive, negative], ignore_index=True).pipe(
            lambda df: LRPropensityDS(
                df.drop('observed', axis=1),
                df.observed
            )
        )

        LR_PROPENSITY_TEST = positive.pipe(
            lambda df: LRPropensityDS(
                df.drop('observed', axis=1),
                df.observed
            )
        )

        FMsLR = mylibml.fm.FactorizationMachinesLogisticRegression(K=5, LAMBDA_w=1, LAMBDA_V=1)
        FMsLR.fit(LR_PROPENSITY_TRAIN.X.values, LR_PROPENSITY_TRAIN.y.values)
        y_pred = FMsLR.predict(LR_PROPENSITY_TEST.X.values)
        error = mean_squared_error(y_pred, LR_PROPENSITY_TEST.y.values)
        print(error)

if __name__ == '__main__':
    unittest.main()

