import mylibml
import pandas as pd
import os
from collections import namedtuple
from sklearn.metrics import mean_squared_error
import unittest

class TestMatrixFactorization(unittest.TestCase):
    def test_fit(self):
        path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'resources/coat')
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

        MF = mylibml.mf.MatrixFactorization(K=5, λ=0.01)
        MF.fit(MF_TRAIN.X.values, MF_TRAIN.y.values)
        y_pred = MF.predict(MF_TEST.X.values)
        error = mean_squared_error(y_pred, MF_TEST.y.values)
        print(error)
        print(MF.score(MF_TEST.X.values, MF_TEST.y.values))


if __name__ == '__main__':
    unittest.main()
