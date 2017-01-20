import numpy as np
import time
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.metrics import mean_squared_error

class FactorizationMachines(BaseEstimator, RegressorMixin):
    def __init__(
            self, K=40, LAMBDA_w=0.001, LAMBDA_V=0.001,
            ETA=0.001, BETA1=0.9, BETA2=0.999, EPS=10e-8,
            THRESHOLD=0.99, LOOP=50, VERBOSE=True):
        self.K = K
        self.LAMBDA_w = LAMBDA_w
        self.LAMBDA_V = LAMBDA_V
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
            'LAMBDA_w': self.LAMBDA_w,
            'LAMBDA_V': self.LAMBDA_V
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def predict(self, X):
        w, V = self.coef
        return np.array([self._predict(x, w, V) for x in X])

    def _predict(self, x, w, V):
        prediction = np.dot(w, x) + 1/2*np.sum(np.square(np.dot(V.T, x)) - np.dot(np.square(V.T), np.square(x)))
        if np.isnan(prediction):
            if self.VERBOSE: print('y_hat is nan.')
            raise RuntimeError()
        return prediction

    def fit(self, X, y):
        fit_start = time.time()
        saturation_counter = 0
        N, D = X.shape
        w = np.random.rand(D)
        V = np.random.rand(D, self.K)
        self.coef = w, V
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
                if self.VERBOSE: print('LOOP{0}: '.format(loop_index), end='')
                for n in range(N):
                    if self.VERBOSE and n % int(N / 10) == 0: print('{0}%...'.format(int(100 * n / N)), end='')
                    y_pred = self._predict(X[n], w, V)
                    e = y_pred - y[n]
                    beta1t = beta1t * self.BETA1
                    beta2t = beta2t * self.BETA2
                    g_w = e*X[n] + self.LAMBDA_w*w
                    m_w = self.BETA1*m_w + (1-self.BETA1)*g_w
                    v_w = self.BETA2*v_w + (1-self.BETA2)*np.square(g_w)
                    mhatt_w = m_w/(1-beta1t)
                    vhatt_w = v_w/(1-beta2t)
                    w = w - self.ETA*mhatt_w/(np.sqrt(vhatt_w)+self.EPS)
                    mask = np.where(X[n] != 0)[0]
                    g_V = e*(X[n][mask][:,np.newaxis]*np.dot(V[mask].T, X[n][mask]) - V[mask]*np.square(X[n][mask][:,np.newaxis])) + self.LAMBDA_V*V[mask]
                    m_V[mask] = self.BETA1*m_V[mask] + (1-self.BETA1)*g_V
                    v_V[mask] = self.BETA2*v_V[mask] + (1-self.BETA2)*np.square(g_V)
                    mhatt_V = m_V[mask]/(1-beta1t)
                    vhatt_V = v_V[mask]/(1-beta2t)
                    V[mask] = V[mask] - self.ETA*mhatt_V/(np.sqrt(vhatt_V)+self.EPS)
                self.coef = w, V
                error = mean_squared_error(y, self.predict(X))
                if self.VERBOSE:
                    print('100% error=> {0} [{1}(sec/it)]'.format(
                        format(error, '.5f'),
                        format(time.time() - start, '.2f')
                    ))
                if (self.THRESHOLD < error / old_error and error / old_error <= 1) \
                        or (self.THRESHOLD < old_error / error and old_error / error <= 1):
                    saturation_counter += 1
                    if saturation_counter == 3:
                        break
                else:
                    saturation_counter = 0
            print('Finished. error => {0} [K={1}, LAMBDA_w={2}, LAMBDA_V={3} {4}(sec)] '.format(
                format(error, '.5f'), self.K, self.LAMBDA_w, self.LAMBDA_V, format(time.time() - fit_start, '.2f')))
            self.coef = w, V
            return self
        except (KeyboardInterrupt, RuntimeError):
            print('Canceled')
            self.coef = w, V
            return self

class PropensityFactorizationMachines(FactorizationMachines):
    def fit(self, X, y):
        fit_start = time.time()
        saturation_counter = 0
        p = X[:, -1]
        assert((0 <= p).all() and (p <= 1).all()) # p は確率
        X = X[:, :-1]
        N, D = X.shape
        w = np.random.rand(D)
        V = np.random.rand(D, self.K)
        self.coef = w, V
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
                if self.VERBOSE: print('LOOP{0}: '.format(loop_index), end='')
                for n in range(N):
                    if self.VERBOSE and n % int(N / 10) == 0: print('{0}%...'.format(int(100 * n / N)), end='')
                    y_pred = self._predict(X[n], w, V)
                    e = y_pred - y[n]
                    beta1t = beta1t * self.BETA1
                    beta2t = beta2t * self.BETA2
                    g_w = 1/p[n]*(e*X[n] + self.LAMBDA_w*w)
                    m_w = self.BETA1*m_w + (1-self.BETA1)*g_w
                    v_w = self.BETA2*v_w + (1-self.BETA2)*np.square(g_w)
                    mhatt_w = m_w/(1-beta1t)
                    vhatt_w = v_w/(1-beta2t)
                    w = w - self.ETA*mhatt_w/(np.sqrt(vhatt_w)+self.EPS)
                    mask = np.where(X[n] != 0)[0]
                    g_V = 1/p[n]*(e*(X[n][mask][:,np.newaxis]*np.dot(V[mask].T, X[n][mask]) - V[mask]*np.square(X[n][mask][:,np.newaxis])) + self.LAMBDA_V*V[mask])
                    m_V[mask] = self.BETA1*m_V[mask] + (1-self.BETA1)*g_V
                    v_V[mask] = self.BETA2*v_V[mask] + (1-self.BETA2)*np.square(g_V)
                    mhatt_V = m_V[mask]/(1-beta1t)
                    vhatt_V = v_V[mask]/(1-beta2t)
                    V[mask] = V[mask] - self.ETA*mhatt_V/(np.sqrt(vhatt_V)+self.EPS)
                self.coef = w, V
                error = mean_squared_error(y, self.predict(X))
                if self.VERBOSE:
                    print('100% error=> {0} [{1}(sec/it)]'.format(
                        format(error, '.5f'),
                        format(time.time() - start, '.2f')
                    ))
                if (self.THRESHOLD < error / old_error and error / old_error <= 1) \
                        or (self.THRESHOLD < old_error / error and old_error / error <= 1):
                    saturation_counter += 1
                    if saturation_counter == 3:
                        break
                else:
                    saturation_counter = 0
            print('Finished. error => {0} [K={1}, LAMBDA_w={2}, LAMBDA_V={3} {4}(sec)] '.format(
                format(error, '.5f'), self.K, self.LAMBDA_w, self.LAMBDA_V, format(time.time() - fit_start, '.2f')))
            self.coef = w, V
            return self
        except (KeyboardInterrupt, RuntimeError):
            print('Canceled')
            self.coef = w, V
            return self

class FactorizationMachinesLogisticRegression(FactorizationMachines, ClassifierMixin):
    def _sigmoid(self, y):
        return 1/(1+1/np.exp(y))

    def _predict(self, x, w, V):
        return self._sigmoid(np.dot(w, x) + 1/2*np.sum(np.square(np.dot(V.T, x)) - np.dot(np.square(V.T), np.square(x))))