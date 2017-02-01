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
        w0, w, V = self.coef
        return np.array([self._predict(x, w0, w, V) for x in X])

    def _predict(self, x, w0, w, V):
        prediction = w0 + np.dot(w, x) + 1/2*np.sum(np.square(np.dot(V.T, x)) - np.dot(np.square(V.T), np.square(x)))
        if np.isnan(prediction):
            if self.VERBOSE: print('prediction is nan', flush=True)
            raise RuntimeError()
        return prediction

    def fit(self, X, y):
        fit_start = time.time()
        saturation_counter = 0
        N, D = X.shape
        w0 = np.random.rand()
        w = np.random.rand(D)
        V = np.random.rand(D, self.K)
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
                r_w0 = self.LAMBDA_w/N * w0
                r_w = self.LAMBDA_w/N *w
                r_V = self.LAMBDA_V/N *V
                for it, n in enumerate(np.random.permutation(range(N))):
                    if self.VERBOSE and it % int(N / 10) == 0: print('{0}%...'.format(int(100 * it / N)), end='', flush=True)
                    y_pred = self._predict(X[n], w0, w, V)
                    e = y_pred - y[n]
                    beta1t = beta1t * self.BETA1
                    beta2t = beta2t * self.BETA2
                    g_w0 = e + r_w0
                    m_w0 = self.BETA1*m_w0 + (1-self.BETA1)*g_w0
                    v_w0 = self.BETA2*v_w0 + (1-self.BETA2)*np.square(g_w0)
                    w0 = w0 - self.ETA*(m_w0/(1-beta1t))/(np.sqrt(v_w0/(1-beta2t))+self.EPS)
                    g_w = e*X[n] + r_w
                    m_w = self.BETA1*m_w + (1-self.BETA1)*g_w
                    v_w = self.BETA2*v_w + (1-self.BETA2)*np.square(g_w)
                    w = w - self.ETA*(m_w/(1-beta1t))/(np.sqrt(v_w/(1-beta2t))+self.EPS)
                    g_V = e*(X[n][:,np.newaxis]*np.dot(V.T, X[n]) - V*np.square(X[n][:,np.newaxis])) + r_V
                    m_V = self.BETA1*m_V + (1-self.BETA1)*g_V
                    v_V = self.BETA2*v_V + (1-self.BETA2)*np.square(g_V)
                    V = V - self.ETA*(m_V/(1-beta1t))/(np.sqrt(v_V/(1-beta2t))+self.EPS)
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
            print('Finished.', end='', flush=True)
            print('error => {0} [K={1}, LAMBDA_w={2}, LAMBDA_V={3} {4}(sec)] '.format(
                format(error, '.5f'), self.K, self.LAMBDA_w, self.LAMBDA_V, format(time.time() - fit_start, '.2f')), flush=True)
            self.coef = w0, w, V
            return self
        except (KeyboardInterrupt, RuntimeError):
            print('Cancelled.', end='', flush=True)
            print('error => {0} [K={1}, LAMBDA_w={2}, LAMBDA_V={3} {4}(sec)] '.format(
                format(error, '.5f'), self.K, self.LAMBDA_w, self.LAMBDA_V, format(time.time() - fit_start, '.2f')), flush=True)
            self.coef = w0, w, V
            return self

class PropensityFactorizationMachines(FactorizationMachines):
    def predict(self, X):
        p = X[:, -1]
        assert((0 <= p).all() and (p <= 1).all()) # p は確率
        X = X[:, :-1]
        w0, w, V = self.coef
        return np.array([self._predict(x, w0, w, V) for x in X])

    def fit(self, X, y):
        fit_start = time.time()
        saturation_counter = 0
        p = X[:, -1]
        assert((0 <= p).all() and (p <= 1).all()) # p は確率
        p = p + 1e-06
        X = X[:, :-1]
        N, D = X.shape
        w0 = np.random.rand()
        w = np.random.rand(D)
        V = np.random.rand(D, self.K)
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
                ips_mean = (1/p).mean()
                r_w0 = self.LAMBDA_w*ips_mean/N*w0
                r_w = self.LAMBDA_w*ips_mean/N*w
                r_V = self.LAMBDA_V*ips_mean/N*V
                for it, n in enumerate(np.random.permutation(range(N))):
                    if self.VERBOSE and it % int(N / 10) == 0: print('{0}%...'.format(int(100 * it / N)), end='', flush=True)
                    y_pred = self._predict(X[n], w0, w, V)
                    e = 1/p[n] * (y_pred - y[n])
                    beta1t = beta1t * self.BETA1
                    beta2t = beta2t * self.BETA2
                    g_w0 = e + r_w0
                    m_w0 = self.BETA1*m_w0 + (1-self.BETA1)*g_w0
                    v_w0 = self.BETA2*v_w0 + (1-self.BETA2)*np.square(g_w0)
                    w0 = w0 - self.ETA*(m_w0/(1-beta1t))/(np.sqrt(v_w0/(1-beta2t))+self.EPS)

                    g_w = e*X[n] + r_w
                    m_w = self.BETA1*m_w + (1-self.BETA1)*g_w
                    v_w = self.BETA2*v_w + (1-self.BETA2)*np.square(g_w)
                    w = w - self.ETA*(m_w/(1-beta1t))/(np.sqrt(v_w/(1-beta2t))+self.EPS)
                    g_V = e*(X[n][:,np.newaxis]*np.dot(V.T, X[n]) - V*np.square(X[n][:,np.newaxis])) + r_V
                    m_V = self.BETA1*m_V + (1-self.BETA1)*g_V
                    v_V = self.BETA2*v_V + (1-self.BETA2)*np.square(g_V)
                    V = V - self.ETA*(m_V/(1-beta1t))/(np.sqrt(v_V/(1-beta2t))+self.EPS)
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
            print('Finished.', end='', flush=True)
            print('error => {0} [K={1}, LAMBDA_w={2}, LAMBDA_V={3} {4}(sec)] '.format(
                format(error, '.5f'), self.K, self.LAMBDA_w, self.LAMBDA_V, format(time.time() - fit_start, '.2f')), flush=True)
            self.coef = w0, w, V
            return self
        except (KeyboardInterrupt, RuntimeError):
            print('Cancelled.', end='', flush=True)
            print('error => {0} [K={1}, LAMBDA_w={2}, LAMBDA_V={3} {4}(sec)] '.format(
                format(error, '.5f'), self.K, self.LAMBDA_w, self.LAMBDA_V, format(time.time() - fit_start, '.2f')), flush=True)
            self.coef = w0, w, V
            return self

class FactorizationMachinesLogisticRegression(BaseEstimator, ClassifierMixin):
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

    def _sigmoid(self, y):
        return 1/(1+np.exp(-y))

    def predict(self, X):
        w0, w, V = self.coef
        return np.array([self._predict(x, w0, w, V) for x in X])

    def _predict(self, x, w0, w, V):
        prediction = self._sigmoid(w0 + np.dot(w, x) + 1/2*np.sum(np.square(np.dot(V.T, x)) - np.dot(np.square(V.T), np.square(x))))
        if np.isnan(prediction):
            if self.VERBOSE: print('prediction is nan', flush=True)
            raise RuntimeError()
        return prediction

    def fit(self, X, y):
        fit_start = time.time()
        saturation_counter = 0
        N, D = X.shape
        w0 = np.random.rand()
        w = np.random.rand(D)
        V = np.random.rand(D, self.K)
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
                r_w0 = self.LAMBDA_w/N * w0
                r_w = self.LAMBDA_w/N *w
                r_V = self.LAMBDA_V/N *V
                for it, n in enumerate(np.random.permutation(range(N))):
                    if self.VERBOSE and it % int(N / 10) == 0: print('{0}%...'.format(int(100 * it / N)), end='', flush=True)
                    y_pred = self._predict(X[n], w0, w, V)
                    e = y_pred - y[n]
                    beta1t = beta1t * self.BETA1
                    beta2t = beta2t * self.BETA2
                    g_w0 = e + r_w0
                    m_w0 = self.BETA1*m_w0 + (1-self.BETA1)*g_w0
                    v_w0 = self.BETA2*v_w0 + (1-self.BETA2)*np.square(g_w0)
                    w0 = w0 - self.ETA*(m_w0/(1-beta1t))/(np.sqrt(v_w0/(1-beta2t))+self.EPS)
                    g_w = e*X[n] + r_w
                    m_w = self.BETA1*m_w + (1-self.BETA1)*g_w
                    v_w = self.BETA2*v_w + (1-self.BETA2)*np.square(g_w)
                    w = w - self.ETA*(m_w/(1-beta1t))/(np.sqrt(v_w/(1-beta2t))+self.EPS)
                    g_V = e*(X[n][:,np.newaxis]*np.dot(V.T, X[n]) - V*np.square(X[n][:,np.newaxis])) + r_V
                    m_V = self.BETA1*m_V + (1-self.BETA1)*g_V
                    v_V = self.BETA2*v_V + (1-self.BETA2)*np.square(g_V)
                    V = V - self.ETA*(m_V/(1-beta1t))/(np.sqrt(v_V/(1-beta2t))+self.EPS)
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
            print('Finished.', end='', flush=True)
            print('error => {0} [K={1}, LAMBDA_w={2}, LAMBDA_V={3} {4}(sec)] '.format(
                format(error, '.5f'), self.K, self.LAMBDA_w, self.LAMBDA_V, format(time.time() - fit_start, '.2f')), flush=True)
            self.coef = w0, w, V
            return self
        except (KeyboardInterrupt, RuntimeError):
            print('Cancelled.', end='', flush=True)
            print('error => {0} [K={1}, LAMBDA_w={2}, LAMBDA_V={3} {4}(sec)] '.format(
                format(error, '.5f'), self.K, self.LAMBDA_w, self.LAMBDA_V, format(time.time() - fit_start, '.2f')), flush=True)
            self.coef = w0, w, V
            return self
