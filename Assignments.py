import numpy as np
from scipy.special import digamma
import tensorflow as tf


class Assignments:
    def __init__(self, pi, psi, rho, N, G, K, L, T):
        self.pi = pi
        self.psi = psi
        self.rho = rho

        self.N = N
        self.G = G
        self.K = K
        self.L = L
        self.T = T

        Phi = np.random.random((N, K))
        self.Phi = Phi / Phi.sum(1)[:, None]

        Lambda = np.random.rand(G, L)
        self.Lambda = Lambda / Lambda.sum(1)[:, None]

        Gamma = np.ones((Lambda.shape[1], 2))
        Gamma[:, 1] = 0
        self.Gamma = Gamma

    def likelihood(self):
        return (np.log(self.pi) * self.Phi).sum() \
                + (np.log(self.psi) * self.Lambda).sum()

    def entropy(self):
        return multinomial_entropy(self.Phi) + multinomial_entropy(self.Lambda)

    def compute_weights(self):
        weights = np.einsum('nk,gl->klng', self.Phi,
                            self.Lambda * self.Gamma[:, 0])
        weights = weights.reshape(self.K * self.L, -1).T
        return weights

    def update_assignments(self, m, X, Y):
        """
        BE AWARE X and Y are the full data with NANS (DONT MASK)
        """
        densities = np.zeros((self.K, self.L, self.N, self.G))
        for t in range(self.T):
            densities_t = m.expected_density(
                X[:, :, t].reshape(-1, 1),
                Y[:, :, t].reshape(-1, 1)).T
            densities += np.nan_to_num(densities_t.reshape(
                self.K, self.L, self.N, self.G))

        for _ in range(10):
            Lambda_old = self.Lambda.copy()

            # sample assignment update
            logPhi = np.nansum(densities[:self.K]
                               * self.Lambda.T[None, :, None, :],
                               axis=(1, 3)).T + np.log(self.pi)

            self.Phi = np.exp(logPhi - logsumexp(logPhi)[:, None])

            # gene assignment update
            logLambda = np.nansum(densities[:self.K]
                                  * self.Phi.T[:, None, :, None],
                                  axis=(0, 2)).T
            logLambda = logLambda + np.log(self.psi)
            self.Lambda = np.exp(logLambda - logsumexp(logLambda)[:, None])

            # local/global gene cluster
            if np.allclose(self.Lambda, Lambda_old):
                break

        self.pi = self.Phi.sum(axis=0) / self.Phi.sum()
        self.psi = self.Lambda.sum(0) / self.Lambda.sum()


class GlobalAssignments:
    def __init__(self, pi, psi, rho, N, G, K, L, T):
        self.pi = pi
        self.psi = psi
        self.rho = rho

        self.N = N
        self.G = G
        self.K = K
        self.L = L
        self.T = T

        Phi = np.random.random((N, K))
        self.Phi = Phi / Phi.sum(1)[:, None]

        Lambda = np.random.rand(G, L)
        self.Lambda = Lambda / Lambda.sum(1)[:, None]

        Gamma = np.ones((Lambda.shape[1], 2)) / 2
        self.Gamma = Gamma

    def likelihood(self):
        return (np.log(self.pi) * self.Phi).sum() \
                + (np.log(self.psi) * self.Lambda).sum() \
                + (np.log(self.rho) * self.Gamma).sum()

    def entropy(self):
        return multinomial_entropy(self.Phi) \
            + multinomial_entropy(self.Lambda) \
            + multinomial_entropy(self.Gamma)

    def compute_weights(self):
        weights = np.einsum('nk,gl->klng', self.Phi,
                            self.Lambda * self.Gamma[:, 0])
        weights = weights.reshape(self.K * self.L, -1).T

        global_weights = np.einsum(
            'na,gl->alng', np.ones(self.N).reshape(-1, 1),
            self.Lambda * self.Gamma[:, 1])
        global_weights = global_weights.reshape(self.L, -1).T
        weights = np.concatenate([weights, global_weights], 1)
        return weights

    def update_assignments(self, m, X, Y):
        """
        BE AWARE X and Y are the full data with NANS (DONT MASK)
        """
        densities = m.expected_density(X.reshape(-1, 1), Y.reshape(-1, 1)).T
        densities = densities.reshape(self.K + 1, self.L, self.N, self.G, self.T)

        for _ in range(10):
            Lambda_old = self.Lambda.copy()

            # sample assignment update
            logPhi = np.nansum(densities[:self.K]
                               * self.Lambda.T[None, :, None, :, None]
                               * self.Gamma[:, 0][None, :, None, None, None],
                               axis=(1, 3, 4)).T + np.log(self.pi)

            self.Phi = np.exp(logPhi - logsumexp(logPhi)[:, None])

            # gene assignment update
            logLambda = np.nansum(densities[:self.K]
                                  * self.Phi.T[:, None, :, None, None]
                                  * self.Gamma[:, 0][None, :, None, None, None],
                                  axis=(0, 2, 4)).T
            logLambda = logLambda + np.nansum(
                densities[self.K] * self.Gamma[:, 1][:, None, None, None],
                axis=(1, 3)).T
            logLambda = logLambda + np.log(self.psi)
            self.Lambda = np.exp(logLambda - logsumexp(logLambda)[:, None])

            # local/global gene cluster
            logGamma = np.zeros((self.L, 2))
            logGamma[:, 0] = np.nansum(
                densities[:self.K] * self.Phi.T[:, None, :, None, None]
                * self.Lambda.T[None, :, None, :, None], axis=(0, 2, 3, 4))

            logGamma[:, 1] = np.nansum(
                densities[self.K] * self.Lambda.T[:, None, :, None], axis=(1, 2, 3))
            logGamma = logGamma + np.log(self.rho)
            Gamma = np.exp(logGamma - logsumexp(logGamma)[:, None])
            Gamma = (Gamma + 0.001)
            self.Gamma = Gamma / Gamma.sum(axis=1)[:, None]

            # local/global gene cluster
            if np.allclose(self.Lambda, Lambda_old):
                break

        self.pi = self.Phi.sum(axis=0) / self.Phi.sum()
        self.psi = self.Lambda.sum(0) / self.Lambda.sum()


class DPAssignments:
    def __init__(self, pi, alpha, N, G, K, L, T):
        self.pi = pi

        self.alpha = alpha
        self.psi = np.ones(L) / L

        self.N = N
        self.G = G
        self.K = K
        self.L = L
        self.T = T

        Phi = np.random.random((N, K))
        self.Phi = Phi / Phi.sum(1)[:, None]

        Lambda = np.random.choice(2, (G, L))
        self.Lambda = Lambda / Lambda.sum(1)[:, None]

        Gamma = np.ones((Lambda.shape[1], 2))
        Gamma[:, 1] = 0
        self.Gamma = Gamma

    def compute_weights(self):
        weights = np.einsum('nk,gl->klng',
                            self.Phi, self.Lambda * self.Gamma[:, 0])
        weights = weights.reshape(self.K * self.L, -1).T
        return weights

    def compute_weights_sparse(self, weight_idx):
        tensors = []
        for k in range(self.K):
            for l in range(self.L):
                entries = np.outer(self.Phi[:, k],
                                   self.Lambda[:, l]).flatten()[weight_idx]
                indices = np.argwhere(~np.isclose(entries[None, :], 0))
                sparse_entries = entries[~np.isclose(entries, 0)]
                tensors.append(tf.SparseTensor(indices, sparse_entries,
                               dense_shape=[1, entries.size]))
        return tf.sparse_concat(0, tensors)

    def compute_weights_individual(self, idx):
        l = idx % self.L
        k = int(idx / self.L)
        return np.outer(self.Phi[:, k], self.Lambda[:, l]).flatten()

    def likelihood(self):
        return 0

    def entropy(self):
        return multinomial_entropy(self.Phi) + multinomial_entropy(self.Lambda)

    def update_assignments(self, m, X, Y):
        """
        BE AWARE X and Y are the full data with NANS (DONT MASK)
        """
        densities = m.expected_density(X.reshape(-1, 1), Y.reshape(-1, 1)).T
        densities = densities.reshape(self.K, self.L, self.N, self.G, self.T)

        for _ in range(10):
            Lambda_old = self.Lambda.copy()

            # sample assignment update
            logPhi = np.nansum(densities[:self.K]
                               * self.Lambda.T[None, :, None, :, None],
                               axis=(1, 3, 4)).T + np.log(self.pi)

            self.Phi = np.exp(logPhi - logsumexp(logPhi)[:, None])

            # gene assignment update
            log_psi = _stick_breaking(self.Lambda, self.alpha)
            logLambda = np.nansum(densities[:self.K]
                                  * self.Phi.T[:, None, :, None, None],
                                  axis=(0, 2, 4)).T
            logLambda = logLambda + log_psi
            new_Lambda = np.exp(logLambda - logsumexp(logLambda)[:, None])
            if np.any(np.isnan(new_Lambda)):
                new_Lambda = Lambda_old
            self.Lambda = new_Lambda
            self.psi = np.exp(log_psi)

            # local/global gene cluster
            if np.allclose(self.Lambda, Lambda_old):
                break

def logsumexp(x):
    """Numerically stable log(sum(exp(x)))"""
    max_x = np.max(x, axis=1)
    return max_x + np.log(np.sum(np.exp(x - max_x[:, np.newaxis]), axis=1))


def multinomial_entropy(p):
    return -1 * np.nansum(p * np.log(p))


def _stick_breaking(Z, concentration):
    """
    return expected values of stick lengths
    alpha concentration parameter
    Z [N, T] expected assignments on truncated DP
    """
    alpha = Z.sum(0)
    beta = np.flip(np.cumsum(np.flip(alpha, 0)), 0) - alpha

    alpha = alpha + 1
    beta = beta + concentration

    digamma_alpha = digamma(alpha)
    digamma_beta = digamma(beta)
    digamma_alpha_beta = digamma(alpha + beta)

    lnv = digamma_alpha - digamma_alpha_beta
    ln1_v = digamma_beta - digamma_alpha_beta

    log_psi = lnv + np.cumsum(ln1_v) - ln1_v
    return log_psi
