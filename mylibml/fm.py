import numpy as np
import time
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.metrics import mean_squared_error
from mylibml.metrics import propensity_scored_mse

class FactorizationMachines(BaseEstimator, RegressorMixin):
    def __init__(
            self, K=40, λ=0.001, σ=0.001,
            ETA=0.001, BETA1=0.9, BETA2=0.999, EPS=10e-8,
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
        return 1/mean_squared_error(y, y_pred)

    def predict(self, X):
        X, α = self.preprocess(X)
        w0, w, V = self.coef
        return np.array([self._predict(x, w0, w, V) for x in X])

    def _predict(self, x, w0, w, V):
        mask = np.where(x != 0)[0]
        prediction = w0 + np.dot(w[mask], x[mask]) + 1/2*np.sum(np.square(np.dot(V[mask].T, x[mask])) - np.dot(np.square(V[mask].T), np.square(x[mask])))
        if np.isnan(prediction):
            if self.VERBOSE: print('prediction is nan', flush=True)
            raise RuntimeError()
        return prediction

    def preprocess(self, X):
        N, D = X.shape
        α = np.ones(N)
        return X, α

    def fit(self, X, y, w0=None, w=None, V=None):
        fit_start = time.time()
        saturation_counter = 0
        X, α = self.preprocess(X)
        N, D = X.shape
        w0 = 0 if w0 is None else w0
        w = np.zeros(D) if w is None else w
        V = np.random.normal(0, self.σ, (D, self.K)) if V is None else V
        if w.shape[0] != D or V.shape != (D, self.K):
            raise ValueError()
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
                r_w0 = λ*w0
                r_w = λ*w
                r_V = λ*V
                for it, n in enumerate(np.random.permutation(range(N))):
                    if self.VERBOSE and it % int(N / 10) == 0: print('{0}%...'.format(int(100 * it / N)), end='', flush=True)
                    y_pred = self._predict(X[n], w0, w, V)
                    e = α[n] * (y_pred - y[n])
                    beta1t = beta1t * self.BETA1
                    beta2t = beta2t * self.BETA2
                    g_w0 = e + r_w0
                    m_w0 = self.BETA1*m_w0 + (1-self.BETA1)*g_w0
                    v_w0 = self.BETA2*v_w0 + (1-self.BETA2)*np.square(g_w0)
                    w0 = w0 - self.ETA*(m_w0/(1-beta1t))/(np.sqrt(v_w0/(1-beta2t))+self.EPS)
                    mask = np.where(X[n] != 0)[0]
                    g_w = e*X[n][mask] + r_w[mask]
                    m_w[mask] = self.BETA1*m_w[mask] + (1-self.BETA1)*g_w
                    v_w[mask] = self.BETA2*v_w[mask] + (1-self.BETA2)*np.square(g_w)
                    w[mask] = w[mask] - self.ETA*(m_w[mask]/(1-beta1t))/(np.sqrt(v_w[mask]/(1-beta2t))+self.EPS)
                    g_V = e*(X[n][:,np.newaxis][mask]*np.dot(V[mask].T, X[n][mask]) - V[mask]*np.square(X[n][:,np.newaxis][mask])) + r_V[mask]
                    m_V[mask] = self.BETA1*m_V[mask] + (1-self.BETA1)*g_V
                    v_V[mask] = self.BETA2*v_V[mask] + (1-self.BETA2)*np.square(g_V)
                    V[mask] = V[mask] - self.ETA*(m_V[mask]/(1-beta1t))/(np.sqrt(v_V[mask]/(1-beta2t))+self.EPS)
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
            print('error => {0} [K={1}, λ={2}, {3}(sec)] '.format(
                format(error, '.5f'), self.K, self.λ, format(time.time() - fit_start, '.2f')), flush=True)
            self.coef = w0, w, V
            return self
        except (KeyboardInterrupt, RuntimeError):
            print('Cancelled.', end='', flush=True)
            print('error => {0} [K={1}, λ={2}, {3}(sec)] '.format(
                format(error, '.5f'), self.K, self.λ, format(time.time() - fit_start, '.2f')), flush=True)
            self.coef = w0, w, V
            return self

class PropensityScoredFactorizationMachines(FactorizationMachines):
    def preprocess(self, X):
        α = 1/X[:, -1]
        if not ((0 <= α).all() and (α <= 1).all()): raise ValueError() # α は確率
        X = X[:, :-1]
        return X, α

class FactorizationMachinesLogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(
            self, K=40, λ=0.001, σ=0.001,
            ETA=0.001, BETA1=0.9, BETA2=0.999, EPS=10e-8,
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

    def _sigmoid(self, y):
        return 1/(1+np.exp(-y))

    def predict(self, X):
        w0, w, V = self.coef
        return np.array([np.round(self._predict(x, w0, w, V)) for x in X])

    def predict_proba(self, X):
        w0, w, V = self.coef
        return np.array([self._predict(x, w0, w, V) for x in X])

    def _predict(self, x, w0, w, V):
        mask = np.where(x != 0)[0]
        prediction = self._sigmoid(w0 + np.dot(w[mask], x[mask]) + 1/2*np.sum(np.square(np.dot(V[mask].T, x[mask])) - np.dot(np.square(V[mask].T), np.square(x[mask]))))
        if np.isnan(prediction):
            if self.VERBOSE: print('prediction is nan', flush=True)
            raise RuntimeError()
        return prediction

    def fit(self, X, y, w0=None, w=None, V=None):
        fit_start = time.time()
        saturation_counter = 0
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
                r_w0 = self.λ/N * w0
                r_w = self.λ/N *w
                r_V = self.λ/N *V
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
                    mask = np.where(X[n] != 0)[0]
                    g_w = e*X[n][mask] + r_w[mask]
                    m_w[mask] = self.BETA1*m_w[mask] + (1-self.BETA1)*g_w
                    v_w[mask] = self.BETA2*v_w[mask] + (1-self.BETA2)*np.square(g_w)
                    w[mask] = w[mask] - self.ETA*(m_w[mask]/(1-beta1t))/(np.sqrt(v_w[mask]/(1-beta2t))+self.EPS)
                    g_V = e*(X[n][:,np.newaxis][mask]*np.dot(V[mask].T, X[n][mask]) - V[mask]*np.square(X[n][:,np.newaxis][mask])) + r_V[mask]
                    m_V[mask] = self.BETA1*m_V[mask] + (1-self.BETA1)*g_V
                    v_V[mask] = self.BETA2*v_V[mask] + (1-self.BETA2)*np.square(g_V)
                    V[mask] = V[mask] - self.ETA*(m_V[mask]/(1-beta1t))/(np.sqrt(v_V[mask]/(1-beta2t))+self.EPS)
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
            print('error => {0} [K={1}, λ={2} {3}(sec)] '.format(
                format(error, '.5f'), self.K, self.λ, format(time.time() - fit_start, '.2f')), flush=True)
            self.coef = w0, w, V
            return self
        except (KeyboardInterrupt, RuntimeError):
            print('Cancelled.', end='', flush=True)
            print('error => {0} [K={1}, λ={2} {3}(sec)] '.format(
                format(error, '.5f'), self.K, self.λ, format(time.time() - fit_start, '.2f')), flush=True)
            self.coef = w0, w, V
            return self
