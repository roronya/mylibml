import numpy as np
import time
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error
from mymllib.metrics import propensity_scored_mse
import os

class BaseMatrixFactorization(BaseEstimator):
    def __init__(
            self, K=40, λ=0.001, σ=0.001, ETA=0.001, BETA1=0.9, BETA2=0.999, EPS=10e-8,
            THRESHOLD=0.99, LOOP=50, VERBOSE=True):
        self.K = K
        self.λ = λ
        self.σ = σ
        self.ETA = ETA
        self.BETA1 = BETA1
        self.BETA2 = BETA2
        self.EPS = EPS
        self.THRESHOLD = THRESHOLD
        self.LOOP = LOOP
        self.VERBOSE = VERBOSE
        self.coef = None

    def get_params(self, deep=True):
        return {
            'K': self.K,
            'λ': self.λ
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
        return 1/mean_squared_error(y_pred, y)

    def predict(self, X):
        X, α = self.preprocess(X)
        w0, w, V = self.coef
        return np.array([self._predict(x, w0, w, V) for x in X])

    def _predict(self, x, w0, w, V):
        i, j = tuple(np.where(x == 1)[0])
        prediction = w0 + w[i] + w[j] + np.dot(V[i], V[j])
        if np.isnan(prediction):
            print('prediction is nan', flush=True)
            raise RuntimeError()
        return prediction

    def preprocess(self, X):
        N, D = X.shape
        α = np.ones(N)
        if not (X.sum(axis=1) == np.array([2]*N)).all(): raise ValueError('X must be 1 in only 2 on every line.')
        return X, α

    def fit(self, X, y, w0=None, w=None, V=None):
        fit_start = time.time()
        saturation_counter = 0
        X, α = self.preprocess(X)
        N, D = X.shape
        w0 = 0 if w0 is None else w0
        w = np.zeros(D) if w is None else w
        V = np.random.normal(0, self.σ, (D, self.K)) if V is None else V
        self.coef = w0, w, V
        m_w0 = 0
        v_w0 = 0
        m_w = np.zeros(D)
        v_w = np.zeros(D)
        m_V = np.zeros((D, self.K))
        v_V = np.zeros((D, self.K))
        beta1t, beta2t = 1, 1
        error = np.inf
        try:
            for loop_index in range(self.LOOP):
                old_error = error
                start = time.time()
                if self.VERBOSE: print('LOOP{0}: '.format(loop_index), end='', flush=True)
                λ = self.λ*α.mean()
                r_V = λ*V
                for it, n in enumerate(np.random.permutation(range(N))):
                    if self.VERBOSE and it % int(N / 10) == 0: print('{0}%...'.format(int(100 * it / N)), end='', flush=True)
                    beta1t = beta1t*self.BETA1
                    beta2t = beta2t*self.BETA2

                    y_pred = self._predict(X[n], w0, w, V)
                    e = α[n] * (y_pred - y[n])
                    g_w0 = e
                    m_w0 = self.BETA1*m_w0 + (1-self.BETA1)*g_w0
                    v_w0 = self.BETA2*v_w0 + (1-self.BETA2)*np.square(g_w0)
                    mhatt_w0 = m_w0/(1-beta1t)
                    vhatt_w0 = v_w0/(1-beta2t)
                    w0 = w0 - self.ETA*mhatt_w0/(np.sqrt(vhatt_w0)+self.EPS)

                    i, j = tuple(np.where(X[n] != 0)[0])
                    g_wi = e*X[n][i]
                    m_w[i] = self.BETA1*m_w[i] + (1-self.BETA1)*g_wi
                    v_w[i] = self.BETA2*v_w[i] + (1-self.BETA2)*np.square(g_wi)
                    w[i] = w[i] - self.ETA*(m_w[i]/1-beta1t)/(np.sqrt(v_w[i]/(1-beta2t))+self.EPS)

                    g_wj = e*X[n][j]
                    m_w[j] = self.BETA1*m_w[j] + (1-self.BETA1)*g_wj
                    v_w[j] = self.BETA2*v_w[j] + (1-self.BETA2)*np.square(g_wj)
                    w[j] = w[j] - self.ETA*(m_w[j]/1-beta1t)/(np.sqrt(v_w[j]/(1-beta2t))+self.EPS)

                    g_Vi = e*V[j] + r_V[i]
                    m_V[i] = self.BETA1*m_V[i] + (1-self.BETA1)*g_Vi
                    v_V[i] = self.BETA2*v_V[i] + (1-self.BETA2)*np.square(g_Vi)
                    V[i] = V[i] - self.ETA/(np.sqrt(v_V[i]/(1-beta2t))+self.EPS) * (m_V[i]/(1-beta1t))
                    g_Vj = e*V[i] + r_V[j]
                    m_V[j] = self.BETA1*m_V[j] + (1-self.BETA1)*g_Vj
                    v_V[j] = self.BETA2*v_V[j] + (1-self.BETA2)*np.square(g_Vj)
                    V[j] = V[j] - self.ETA/(np.sqrt(v_V[j]/(1-beta2t))+self.EPS) * (m_V[j]/(1-beta1t))

                self.coef = w0, w, V
                y_pred = np.array([self._predict(x, w0, w, V) for x in X])
                error = mean_squared_error(y, y_pred)
                if self.VERBOSE:
                    print('100% error=> {0} [{1}(sec/it)]'.format(
                        format(error, '.5f'),
                        format(time.time() - start, '.2f')
                    ), flush=True)
                if (self.THRESHOLD < error / old_error and error / old_error <= 1) \
                        or (self.THRESHOLD < old_error / error and old_error / error <= 1):
                    saturation_counter += 1
                    if saturation_counter == 3:
                        break
                else:
                    saturation_counter = 0
            print('Finished. error => {0} [K={1}, λ={2}, {3}(sec)] '.format(
                format(error, '.5f'), self.K, self.λ, format(time.time() - fit_start, '.2f')), flush=True)
            self.coef = w0, w, V
            return self
        except (KeyboardInterrupt, RuntimeError):
            print('Canceled', flush=True)
            self.coef = w0, w, V
            return self

    def save(self, filepath='.'):
        w0, w, V = self.coef
        np.save(os.path.join(filepath, 'w0.csv'), w0)
        np.save(os.path.join(filepath, 'w.csv'), w)
        np.save(os.path.join(filepath, 'V.csv'), V)




class MatrixFactorization(BaseMatrixFactorization, RegressorMixin):
    """
    MatrixFactorization for regressor
    """

class PropensityScoredMatrixFactorization(MatrixFactorization):
    def preprocess(self, X):
        α = 1/X[:, -1]
        if not (0 <= 1/α).all() and (1/α <= 1).all(): raise ValueError('α must be inverse probability.')
        X = X[:, :-1]
        N, D = X.shape
        if not (X.sum(axis=1) == np.array([2]*N)).all(): raise ValueError('X must be 1 in only 2 on every line.')
        return X, α

    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
        return 1/propensity_scored_mse(y, y_pred)
