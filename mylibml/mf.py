import numpy as np
import time
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error

class MatrixFactorization(BaseEstimator, RegressorMixin):
    def __init__(
            self, K=40, LAMBDA=0.001, ETA=0.001, BETA1=0.9, BETA2=0.999, EPS=10e-8,
            THRESHOLD=0.99, LOOP=50, VERBOSE=True):
        self.K = K
        self.LAMBDA = LAMBDA
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
            'LAMBDA': self.LAMBDA
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def predict(self, X):
        N, D = X.shape
        assert((X.sum(axis=1) == np.array([2]*N)).all()) # X は全ての行で2つだけ1が立つ
        V = self.coef
        return np.array([self._predict(x, V) for x in X])

    def _predict(self, x, V):
        i, j = tuple(np.where(x == 1)[0])
        prediction = np.dot(V[i], V[j])
        if np.isnan(prediction):
            print('prediction is nan')
            raise RuntimeError()
        return prediction

    def fit(self, X, y):
        fit_start = time.time()
        saturation_counter = 0
        N, D = X.shape
        assert((X.sum(axis=1) == np.array([2]*N)).all()) # X は全ての行で2つだけ1が立つ
        V = np.random.rand(D, self.K)
        self.coef = V
        m = np.zeros((D, self.K))
        v = np.zeros((D, self.K))
        beta1t, beta2t = 1, 1
        error = np.inf
        try:
            for loop_index in range(self.LOOP):
                start = time.time()
                old_error = error
                if self.VERBOSE: print('LOOP{0}: '.format(loop_index), end='')
                for it, n in enumerate(np.random.permutation(range(N))):
                    if self.VERBOSE and it % int(N / 10) == 0: print('{0}%...'.format(int(100 * it / N)), end='')
                    y_pred = self._predict(X[n], V)
                    e = y_pred - y[n]
                    i, j = tuple(np.where(X[n] == 1)[0])
                    # Adam
                    beta1t = beta1t*self.BETA1
                    beta2t = beta2t*self.BETA2
                    g = e*V[j] + self.LAMBDA*V[i]
                    m[i] = self.BETA1*m[i] + (1-self.BETA1)*g
                    v[i] = self.BETA2*v[i] + (1-self.BETA2)*np.square(g)
                    mhatt = m[i]/(1-beta1t)
                    vhatt = v[i]/(1-beta2t)
                    V[i] = V[i] - self.ETA/(np.sqrt(vhatt)+self.EPS) * mhatt
                    g = e*V[i] + self.LAMBDA*V[j]
                    m[j] = self.BETA1*m[j] + (1-self.BETA1)*g
                    v[j] = self.BETA2*v[j] + (1-self.BETA2)*np.square(g)
                    mhatt = m[j]/(1-beta1t)
                    vhatt = v[j]/(1-beta2t)
                    V[j] = V[j] - self.ETA/(np.sqrt(vhatt)+self.EPS) * mhatt
                self.coef = V
                error = mean_squared_error(y, self.predict(X))
                if self.VERBOSE:
                    print('100% error=> {0} [{1}(sec/it)]'.format(
                        format(error, '.5f'),
                        format(time.time() - start, '.2f')
                    ))
                if (self.THRESHOLD < error / old_error and error / old_error <= 1) \
                        or (self.THRESHOLD < old_error / error and old_error / error <= 1):
                    saturation_counter += 1
                    if saturation_counter == 5:
                        break
                else:
                    saturation_counter = 0
            print('Finished. error => {0} [K={1}, LAMBDA={2}, {3}(sec)] '.format(
                format(error, '.5f'), self.K, self.LAMBDA, format(time.time() - fit_start, '.2f')))
            self.coef = V
            return self
        except (KeyboardInterrupt, RuntimeError):
            print('Cancelled')
            self.coef = V
            return self

class PropensityMatrixFactorization(MatrixFactorization):
    def fit(self, X, y):
        fit_start = time.time()
        saturation_counter = 0
        p = X[:, -1]
        assert((0 <= p).all() and (p <= 1).all()) # p は確率
        X = X[:, :-1]
        N, D = X.shape
        assert((X.sum(axis=1) == np.array([2]*N)).all()) # X は全ての行で2つだけ1が立つ
        V = np.random.rand(D, self.K)
        self.coef = V
        m = np.zeros((D, self.K))
        v = np.zeros((D, self.K))
        beta1t, beta2t = 1, 1
        error = np.inf
        try:
            for loop_index in range(self.LOOP):
                start = time.time()
                old_error = error
                if self.VERBOSE: print('LOOP{0}: '.format(loop_index), end='')
                for it, n in enumerate(np.random.permutation(range(N))):
                    if self.VERBOSE and it % int(N / 10) == 0: print('{0}%...'.format(int(100 * it / N)), end='')
                    y_pred = self._predict(X[n], V)
                    e = y_pred - y[n]
                    i, j = tuple(np.where(X[n] == 1)[0])
                    # Adam
                    beta1t = beta1t*self.BETA1
                    beta2t = beta2t*self.BETA2
                    g = 1/p[n]*(e*V[j] + self.LAMBDA*V[i])
                    m[i] = self.BETA1*m[i] + (1-self.BETA1)*g
                    v[i] = self.BETA2*v[i] + (1-self.BETA2)*np.square(g)
                    mhatt = m[i]/(1-beta1t)
                    vhatt = v[i]/(1-beta2t)
                    V[i] = V[i] - self.ETA/(np.sqrt(vhatt)+self.EPS) * mhatt
                    g = 1/p[n]*(e*V[i] + self.LAMBDA*V[j])
                    m[j] = self.BETA1*m[j] + (1-self.BETA1)*g
                    v[j] = self.BETA2*v[j] + (1-self.BETA2)*np.square(g)
                    mhatt = m[j]/(1-beta1t)
                    vhatt = v[j]/(1-beta2t)
                    V[j] = V[j] - self.ETA/(np.sqrt(vhatt)+self.EPS) * mhatt
                self.coef = V
                error = mean_squared_error(y, self.predict(X))
                if self.VERBOSE:
                    print('100% error=> {0} [{1}(sec/it)]'.format(
                        format(error, '.5f'),
                        format(time.time() - start, '.2f')
                    ))
                if (self.THRESHOLD < error / old_error and error / old_error <= 1) \
                        or (self.THRESHOLD < old_error / error and old_error / error <= 1):
                    saturation_counter += 1
                    if saturation_counter == 5:
                        break
                else:
                    saturation_counter = 0
            print('Finished. error => {0} [K={1}, LAMBDA={2}, {3}(sec)] '.format(
                format(error, '.5f'), self.K, self.LAMBDA, format(time.time() - fit_start, '.2f')))
            self.coef = V
            return self
        except (KeyboardInterrupt, RuntimeError):
            print('Cancelled')
            self.coef = V
            return self
