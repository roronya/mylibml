import mymllib
import pandas as pd
import os
from collections import namedtuple
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import unittest

class TestPropensityMatrixFactorization(unittest.TestCase):
    def test_fit(self):
        path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'resources/coat')

        RAW_TRAIN = pd.read_csv(
            'resources/coat/train.ascii', header=None, delimiter=' '
        )

        VERTICAL_RAW_TRAIN = RAW_TRAIN.reset_index().pipe(
            lambda df: pd.melt(df, id_vars=['index'], value_vars=list(range(300)))
        ).rename(columns={'index': 'user_id', 'variable': 'item_id', 'value': 'rating'}).pipe(
            lambda df: df[df.rating > 0]
        ).reset_index(drop=True)

        TRAIN, VALIDATION = train_test_split(VERTICAL_RAW_TRAIN, test_size=0.2)
        TRAIN = TRAIN.reset_index(drop=True)
        VALIDATION = VALIDATION.reset_index(drop=True)

        RAW_TEST = pd.read_csv(
            os.path.join(path, 'test.ascii'), header=None, delimiter=' '
        )

        TEST = RAW_TEST.reset_index().pipe(
            lambda df: pd.melt(df, id_vars=['index'], value_vars=list(range(300)))
        ).rename(columns={'index': 'user_id', 'variable': 'item_id', 'value': 'rating'}).pipe(
            lambda df: df[df.rating > 0]
        ).reset_index(drop=True)

        MFDS = namedtuple('MFDS', ('X', 'y'))
        MF_TRAIN = TRAIN.pipe(
            lambda df: pd.get_dummies(df[['rating', 'user_id', 'item_id']], columns=['user_id', 'item_id'])
        ).pipe(
            lambda df: MFDS(
                df.drop('rating', axis=1),
                df.rating
            )
        )

        MF_TEST = TEST.pipe(
            lambda df: pd.get_dummies(df[['rating', 'user_id', 'item_id']], columns=['user_id', 'item_id'])
        ).pipe(
            lambda df: MFDS(
                df.drop('rating', axis=1)[MF_TRAIN.X.columns],
                df.rating
            )
        )

        RAW_ITEM = pd.read_csv(
            'resources/coat/item_features.ascii', header=None, delimiter=' '
        )

        ITEM = RAW_ITEM.add_prefix('item_f_').reset_index().rename(
            columns={'index': 'item_id'}
        )

        RAW_USER = pd.read_csv(
            'resources/coat/user_features.ascii', header=None, delimiter=' '
        )

        USER = RAW_USER.add_prefix('user_f_').reset_index().rename(
            columns={'index': 'user_id'}
        )

        NB_PROPENSITY = TRAIN.pipe(
            lambda df: pd.merge(
                df,
                TRAIN.pipe(lambda df: df.groupby('rating').size() / df.shape[0]).rename(
                    'train_rating_prob').reset_index(),
                on='rating'
            )
        ).pipe(
            lambda df: pd.merge(
                df,
                TEST.pipe(lambda df: df.groupby('rating').size() / df.shape[0]).rename(
                    'test_rating_prob').reset_index(),
                on='rating'
            )
        ).assign(
            obs_prob=lambda df: df.shape[0] / (ITEM.shape[0] * USER.shape[0])
        ).assign(
            propensity=lambda df: df.train_rating_prob * df.obs_prob / df.test_rating_prob
        ).pipe(
            lambda df: pd.merge(TRAIN, df, how='left', right_on=['user_id', 'item_id'], left_on=['user_id', 'item_id'])
        )

        MF = mymllib.mf.PropensityScoredMatrixFactorization(K=5, λ=0.1, σ=0.0001)
        MF.fit(MF_TRAIN.X.assign(propensity=NB_PROPENSITY.propensity.values).values, MF_TRAIN.y.values)
        y_pred = MF.predict(MF_TEST.X.assign(propensity=1).values)
        error = mean_squared_error(y_pred, MF_TEST.y.values)
        print(error)


if __name__ == '__main__':
    unittest.main()
