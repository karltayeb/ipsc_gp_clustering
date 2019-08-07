import numpy as np
import tensorflow as tf

from gpflow.models import GPModel
from gpflow import likelihoods
from gpflow import settings
from gpflow.decors import autoflow
from gpflow.decors import params_as_tensors
from gpflow.mean_functions import Zero
from gpflow.params import DataHolder, Parameter, Minibatch
from gpflow import transforms, kullback_leiblers, features
from gpflow.conditionals import conditional
import gpflow

class SplitGPMOld(GPModel):
    """
    Sparse Variational GP regression. The key reference is
    ::
      @inproceedings{titsias2009variational,
        title={Variational learning of inducing variables in
               sparse Gaussian processes},
        author={Titsias, Michalis K},
        booktitle={International Conference onum_i
                   Artificial Intelligence and Statistics},
        pages={567--574},
        year={2009}
      }
    """

    def __init__(self, X, Y, W1, W2, kern, likelihood,
                 idx=None, W1_idx=None, W2_idx=None, feat=None,
                 mean_function=None,
                 num_latent=None,
                 q_diag=False,
                 whiten=True,
                 minibatch_size=None,
                 Z=None,
                 num_data=None,
                 q_mu=None,
                 q_sqrt=None,
                 **kwargs):
        """
        - X is a data matrix, size N x D
        - Y is a data matrix, size N x P
        - kern, likelihood, mean_function are appropriate GPflow objects
        - Z is a matrix of pseudo inputs, size M x D
        - num_latent is the number of latent process to use, default to
          Y.shape[1]
        - q_diag is a boolean. If True, the covariance is approximated by a
          diagonal matrix.
        - whiten is a boolean. If True, we use the whitened representation of
          the inducing points.
        - minibatch_size, if not None, turns on mini-batching with that size.
        - num_data is the total number of observations, default to X.shape[0]
          (relevant when feeding in external minibatches)
        """
        # sort out the X, Y into MiniBatch objects if required.
        num_data = X.shape[0]

        if minibatch_size is None:
            X = DataHolder(X)
            Y = DataHolder(Y)

            if W1_idx is not None:
                W1_idx = DataHolder(W1_idx, fix_shape=True)

            if W2_idx is not None:
                W2_idx = DataHolder(W2_idx, fix_shape=True)
        else:
            X = Minibatch(X, batch_size=minibatch_size, seed=0)
            Y = Minibatch(Y, batch_size=minibatch_size, seed=0)
            
            idx = Minibatch(np.arange(num_data), batch_size=minibatch_size, seed=0, dtype=np.int32)
            if W1_idx is not None:
                W1_idx = Minibatch(
                    W1_idx, batch_size=minibatch_size, seed=0, dtype=np.int32)

            if W2_idx is not None:
                W2_idx = Minibatch(
                    W2_idx, batch_size=minibatch_size, seed=0, dtype=np.int32)

        # init the super class, accept args
        num_latent = W1.shape[1] * W2.shape[1]
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function, num_latent, **kwargs)
        self.num_data = num_data or X.shape[0]
        self.q_diag, self.whiten = q_diag, whiten
        self.feature = features.inducingpoint_wrapper(feat, Z)

        self.idx = idx
        self.W1_idx = W1_idx
        self.W2_idx = W2_idx

        self.K1 = W1.shape[1]
        self.W1 = Parameter(W1, trainable=False, dtype=settings.float_type)
        self.W1_prior = Parameter(np.ones(self.K1) / self.K1, trainable=False)

        self.K2 = W2.shape[1]
        self.W2 = Parameter(W2, trainable=False, dtype=settings.float_type)
        self.W2_prior = Parameter(np.ones(self.K2) / self.K2, trainable=False)

        # init variational parameters
        num_inducing = len(self.feature)
        self._init_variational_parameters(num_inducing, q_mu, q_sqrt, q_diag)

    def _init_variational_parameters(self, num_inducing, q_mu, q_sqrt, q_diag):
        """
        Constructs the mean and cholesky of the covariance of the variational Gaussian posterior.
        If a user passes values for `q_mu` and `q_sqrt` the routine checks if they have consistent
        and correct shapes. If a user does not specify any values for `q_mu` and `q_sqrt`, the routine
        initializes them, their shape depends on `num_inducing` and `q_diag`.
        Note: most often the comments refer to the number of observations (=output dimensions) with P,
        number of latent GPs with L, and number of inducing points M. Typically P equals L,
        but when certain multioutput kernels are used, this can change.
        Parameters
        ----------
        :param num_inducing: int
            Number of inducing variables, typically refered to as M.
        :param q_mu: np.array or None
            Mean of the variational Gaussian posterior. If None the function will initialise
            the mean with zeros. If not None, the shape of `q_mu` is checked.
        :param q_sqrt: np.array or None
            Cholesky of the covariance of the variational Gaussian posterior.
            If None the function will initialise `q_sqrt` with identity matrix.
            If not None, the shape of `q_sqrt` is checked, depending on `q_diag`.
        :param q_diag: bool
            Used to check if `q_mu` and `q_sqrt` have the correct shape or to
            construct them with the correct shape. If `q_diag` is true,
            `q_sqrt` is two dimensional and only holds the square root of the
            covariance diagonal elements. If False, `q_sqrt` is three dimensional.
        """
        q_mu = np.zeros((num_inducing, self.num_latent)) if q_mu is None else q_mu
        self.q_mu = Parameter(q_mu, dtype=settings.float_type)  # M x P

        if q_sqrt is None:
            if self.q_diag:
                self.q_sqrt = Parameter(np.ones((num_inducing, self.num_latent), dtype=settings.float_type),
                                        transform=transforms.positive)  # M x P
            else:
                q_sqrt = np.array([np.eye(num_inducing, dtype=settings.float_type) for _ in range(self.num_latent)])
                self.q_sqrt = Parameter(q_sqrt, transform=transforms.LowerTriangular(num_inducing, self.num_latent))  # P x M x M
        else:
            if q_diag:
                assert q_sqrt.ndim == 2
                self.num_latent = q_sqrt.shape[1]
                self.q_sqrt = Parameter(q_sqrt, transform=transforms.positive)  # M x L/P
            else:
                assert q_sqrt.ndim == 3
                self.num_latent = q_sqrt.shape[0]
                num_inducing = q_sqrt.shape[1]
                self.q_sqrt = Parameter(q_sqrt, transform=transforms.LowerTriangular(num_inducing, self.num_latent))  # L/P x M x M

    @params_as_tensors
    def build_prior_KL(self):
        if self.whiten:
            K = None
        else:
            K = features.Kuu(
                self.feature, self.kern,
                jitter=settings.numerics.jitter_level)  # (P x) x M x M

        return kullback_leiblers.gauss_kl(self.q_mu, self.q_sqrt, K)

    @params_as_tensors
    def build_prior_assignment_KL(self, W1_active, W2_active):
        W1 = tf.gather(self.W1, W1_active)
        W2 = tf.gather(self.W2, W2_active)

        KL = 0

        if True:
            KL += tf.reduce_sum(normalize(W1) * (
                tf.log(normalize(W1)) - tf.log(self.W1_prior)[None])) \

        if True:
            KL += tf.reduce_sum(normalize(W2) * (
                tf.log(normalize(W2)) - tf.log(self.W2_prior)[None]))

        return KL

    @params_as_tensors
    def _build_likelihood(self):
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood. For a derivation of the terms in here, see the associated
        SGPR notebook.
        """
        Y = self.Y
        X = self.X
        idx = self.idx

        W1_idx = self.W1_idx
        W2_idx = self.W2_idx

        if W1_idx is None:
            W1_idx = idx
        if W2_idx is None:
            W2_idx = idx

        if tf.shape(X)[0] != self.num_data:
            W1 = tf.gather(self.W1, W1_idx)
            W1 = tf.reshape(W1, [-1, self.K1])
            W1 = normalize(W1)

            W2 = tf.gather(self.W2, W2_idx)
            W2 = tf.reshape(W2, [-1, self.K2])
            W2 = normalize(W2)

        else:
            W1 = normalize(self.W1)  # N x K1
            if W1_idx is not None:
                W1 = tf.gather(W1, W1_idx)
            W2 = normalize(self.W2)  # N x K2
            if W2_idx is not None:
                W2 = tf.gather(W2, W2_idx)

        W = _expand_W(W1, W2)

       # compute KL
        KL1 = self.build_prior_KL()
        KL2 = self.build_prior_assignment_KL(tf.unique(W1_idx)[0], tf.unique(W2_idx)[0])

        fmean, fvar = self._build_predict(X, full_cov=False, full_output_cov=False)

        # if we need to use quadrature, we need to expand Y to full dimensions
        # otherwise we can get away with not tiling
        if self.likelihood.__class__.variational_expectations \
            == likelihoods.Likelihood.variational_expectations:
            var_exp = self.likelihood.variational_expectations(fmean, fvar, tf.tile(Y, [1, self.K1 * self.K2]))
        else:
            var_exp = self.likelihood.variational_expectations(fmean, fvar, Y)
        scale = tf.cast(self.num_data, settings.float_type) / \
            tf.cast(tf.shape(X)[0], settings.float_type)

        bound = tf.reduce_sum(W * var_exp)
        bound *= scale
        bound -= KL1 + KL2
        return bound

    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False, full_output_cov=False):
        mu, var = conditional(
            Xnew, self.feature, self.kern,
            self.q_mu, q_sqrt=self.q_sqrt, full_cov=full_cov,
            white=self.whiten, full_output_cov=full_output_cov)
        return mu + self.mean_function(Xnew), var

class SplitGPM(GPModel):
    """
    ::
      @inproceedings{titsias2009variational,
        title={Variational learning of inducing variables in
               sparse Gaussian processes},
        author={Titsias, Michalis K},
        booktitle={International Conference onum_i
                   Artificial Intelligence and Statistics},
        pages={567--574},
        year={2009}
      }
    """

    def __init__(self, X, Y, W1, W2, kern, likelihood,
                 feat=None,
                 mean_function=None,
                 num_latent=None,
                 q_diag=False,
                 whiten=True,
                 minibatch_size=None,
                 Z=None,
                 num_data=None,
                 q_mu=None,
                 q_sqrt=None,
                 **kwargs):
        """
        - X is a data matrix, size N x D
        - Y is a data matrix, size N x P
        - kern, likelihood, mean_function are appropriate GPflow objects
        - Z is a matrix of pseudo inputs, size M x D
        - num_latent is the number of latent process to use, default to
          Y.shape[1]
        - q_diag is a boolean. If True, the covariance is approximated by a
          diagonal matrix.
        - whiten is a boolean. If True, we use the whitened representation of
          the inducing points.
        - minibatch_size, if not None, turns on mini-batching with that size.
        - num_data is the total number of observations, default to X.shape[0]
          (relevant when feeding in external minibatches)

        THE LAST TWO COLUMNS OF X ARE W1, W2 idx
        """
        # sort out the X, Y into MiniBatch objects if required.
        num_data = X.shape[0]

        if minibatch_size is None:
            X = DataHolder(X)
            Y = DataHolder(Y)
        else:
            X = Minibatch(X, batch_size=minibatch_size, seed=0)
            Y = Minibatch(Y, batch_size=minibatch_size, seed=0)

        # init the super class, accept args
        num_latent = W1.shape[1] * W2.shape[1]
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function, num_latent, **kwargs)
        self.num_data = num_data or X.shape[0]
        self.q_diag, self.whiten = q_diag, whiten
        self.feature = features.inducingpoint_wrapper(feat, Z)

        self.K1 = W1.shape[1]
        self.W1 = Parameter(W1, trainable=False, dtype=settings.float_type)
        self.W1_prior = Parameter(np.ones(self.K1) / self.K1, trainable=False)

        self.K2 = W2.shape[1]
        self.W2 = Parameter(W2, trainable=False, dtype=settings.float_type)
        self.W2_prior = Parameter(np.ones(self.K2) / self.K2, trainable=False)

        # init variational parameters
        num_inducing = len(self.feature)
        self._init_variational_parameters(num_inducing, q_mu, q_sqrt, q_diag)

    def _init_variational_parameters(self, num_inducing, q_mu, q_sqrt, q_diag):
        """
        Constructs the mean and cholesky of the covariance of the variational Gaussian posterior.
        If a user passes values for `q_mu` and `q_sqrt` the routine checks if they have consistent
        and correct shapes. If a user does not specify any values for `q_mu` and `q_sqrt`, the routine
        initializes them, their shape depends on `num_inducing` and `q_diag`.
        Note: most often the comments refer to the number of observations (=output dimensions) with P,
        number of latent GPs with L, and number of inducing points M. Typically P equals L,
        but when certain multioutput kernels are used, this can change.
        Parameters
        ----------
        :param num_inducing: int
            Number of inducing variables, typically refered to as M.
        :param q_mu: np.array or None
            Mean of the variational Gaussian posterior. If None the function will initialise
            the mean with zeros. If not None, the shape of `q_mu` is checked.
        :param q_sqrt: np.array or None
            Cholesky of the covariance of the variational Gaussian posterior.
            If None the function will initialise `q_sqrt` with identity matrix.
            If not None, the shape of `q_sqrt` is checked, depending on `q_diag`.
        :param q_diag: bool
            Used to check if `q_mu` and `q_sqrt` have the correct shape or to
            construct them with the correct shape. If `q_diag` is true,
            `q_sqrt` is two dimensional and only holds the square root of the
            covariance diagonal elements. If False, `q_sqrt` is three dimensional.
        """
        q_mu = np.zeros((num_inducing, self.num_latent)) if q_mu is None else q_mu
        self.q_mu = Parameter(q_mu, dtype=settings.float_type)  # M x P

        if q_sqrt is None:
            if self.q_diag:
                self.q_sqrt = Parameter(np.ones((num_inducing, self.num_latent), dtype=settings.float_type),
                                        transform=transforms.positive)  # M x P
            else:
                q_sqrt = np.array([np.eye(num_inducing, dtype=settings.float_type) for _ in range(self.num_latent)])
                self.q_sqrt = Parameter(q_sqrt, transform=transforms.LowerTriangular(num_inducing, self.num_latent))  # P x M x M
        else:
            if q_diag:
                assert q_sqrt.ndim == 2
                self.num_latent = q_sqrt.shape[1]
                self.q_sqrt = Parameter(q_sqrt, transform=transforms.positive)  # M x L/P
            else:
                assert q_sqrt.ndim == 3
                self.num_latent = q_sqrt.shape[0]
                num_inducing = q_sqrt.shape[1]
                self.q_sqrt = Parameter(q_sqrt, transform=transforms.LowerTriangular(num_inducing, self.num_latent))  # L/P x M x M

    @params_as_tensors
    def build_prior_KL(self):
        # q_mu KL
        if self.whiten:
            K = None
        else:
            K = features.Kuu(
                self.feature, self.kern,
                jitter=settings.numerics.jitter_level)  # (P x) x M x M
        KL = kullback_leiblers.gauss_kl(self.q_mu, self.q_sqrt, K)

        # assignment KL
        KL += tf.reduce_sum(normalize(self.W1) * (
            tf.log(normalize(self.W1)) - tf.log(self.W1_prior)[None]))
        KL += tf.reduce_sum(normalize(self.W2) * (
            tf.log(normalize(self.W2)) - tf.log(self.W2_prior)[None]))
        return KL

    @params_as_tensors
    def _build_likelihood(self):
        """
        This gives a variational bound on the model likelihood.
        """
        X = self.X
        Y = self.Y

        W1_idx = tf.cast(X[:, -2], tf.int32)
        W2_idx = tf.cast(X[:, -1], tf.int32)

        W1 = tf.nn.softmax(self.W1)  # N x K1
        W1 = tf.gather(W1, W1_idx)
        W1 = tf.reshape(W1, [-1, self.K1])
        
        W2 = tf.nn.softmax(self.W2)  # N x K2
        W2 = tf.gather(W2, W2_idx)
        W2 = tf.reshape(W2, [-1, self.K2])

        # W = _expand_W(W1, W2)

        # Get prior KL.
        KL = self.build_prior_KL()

        # Get conditionals
        fmean, fvar = self._build_predict(X, full_cov=False, full_output_cov=False)

        # if we need to use quadrature, we need to expand Y to full dimensions
        # otherwise we can get away with not tiling
        if self.likelihood.__class__.variational_expectations \
            == gpflow.likelihoods.Likelihood.variational_expectations:
            var_exp = self.likelihood.variational_expectations(fmean, fvar, tf.tile(Y, [1, self.num_latent]))
        else:
            var_exp = self.likelihood.variational_expectations(fmean, fvar, Y)

        var_exp = tf.transpose(tf.reshape(tf.transpose(var_exp), [self.K1, self.K2, -1])) # N x K2 x K1
        # re-scale for minibatch size
        scale = tf.cast(self.num_data, settings.float_type) / tf.cast(tf.shape(X)[0], settings.float_type)
        return tf.reduce_sum(var_exp * W1[:, None, :] * W2[:, :, None]) * scale - KL

    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False, full_output_cov=False):
        mu, var = conditional(
            Xnew, self.feature, self.kern,
            self.q_mu, q_sqrt=self.q_sqrt, full_cov=full_cov,
            white=self.whiten, full_output_cov=full_output_cov)
        return mu + self.mean_function(Xnew), var

    @autoflow((settings.float_type, [None, None]), (settings.float_type, [None, None]))
    @params_as_tensors
    def update_W1(self, X, Y):
        """
        This only works if it gets all of the data associated with an assignment
        pass data to these update functions accordingly.
        """
        W1_idx = tf.cast(X[:, -2], tf.int32)
        W2_idx = tf.cast(X[:, -1], tf.int32)
        fmean, fvar = self._build_predict(
            X, full_cov=False, full_output_cov=False)

        if self.likelihood.__class__.variational_expectations \
            == likelihoods.Likelihood.variational_expectations:
            var_exp = self.likelihood.variational_expectations(fmean, fvar, tf.tile(Y, [1, self.K1 * self.K2]))
        else:
            var_exp = self.likelihood.variational_expectations(fmean, fvar, Y)

        logW = tf.reshape(tf.transpose(var_exp), [self.K1, self.K2, -1])
        # compute new W1
        W2 = tf.gather(normalize(self.W2), W2_idx)
        logW1 = tf.reduce_sum(
            logW * tf.transpose(W2)[None, :, :],
            axis=1)

        # group by index
        num_partitions = self.W1.shape[0]
        logW1_parts = tf.dynamic_partition(
            tf.transpose(logW1), W1_idx, num_partitions=num_partitions)
        logW1 = tf.stack([
            tf.reduce_sum(part, axis=0) for part in logW1_parts])
        logW1 = logW1 + tf.log(self.W1_prior)[None]
        logW1 = logW1 - tf.reduce_logsumexp(logW1, axis=1, keepdims=True)
        return logW1

    @autoflow((settings.float_type, [None, None]), (settings.float_type, [None, None]))
    @params_as_tensors
    def update_W2(self, X, Y):
        W1_idx = tf.cast(X[:, -2], tf.int32)
        W2_idx = tf.cast(X[:, -1], tf.int32)
        fmean, fvar = self._build_predict(
            X, full_cov=False, full_output_cov=False)

        if self.likelihood.__class__.variational_expectations \
            == likelihoods.Likelihood.variational_expectations:
            var_exp = self.likelihood.variational_expectations(fmean, fvar, tf.tile(Y, [1, self.K1 * self.K2]))
        else:
            var_exp = self.likelihood.variational_expectations(fmean, fvar, Y)

        logW = tf.reshape(tf.transpose(var_exp), [self.K1, self.K2, -1])  # K1 x K2 x N

        # compute new W2
        W1 = tf.gather(normalize(self.W1), W1_idx)  # N x K1
        logW2 = tf.reduce_sum(
            logW * tf.transpose(W1)[:, None, :],
            axis=0)  # K2 x N

        # group by index
        num_partitions = self.W2.shape[0]
        logW2_parts = tf.dynamic_partition(
            tf.transpose(logW2), W2_idx, num_partitions=num_partitions)
        logW2 = tf.stack([
            tf.reduce_sum(part, axis=0) for part in logW2_parts])
        logW2 = logW2 + tf.log(self.W2_prior[None])
        logW2 = logW2 - tf.reduce_logsumexp(logW2, axis=1, keepdims=True)
        return logW2

def normalize(W, epsilon=1e-3):
    expW = tf.exp(W) + epsilon
    return expW / tf.reduce_sum(expW, axis=1)[:, None]

def _expand_W(W1, W2):
    K1 = W1.shape[1]
    K2 = W2.shape[1]

    if not isinstance(K1, int):
        K1 = K1.value
        K2 = K2.value

    W = []
    for i in range(K1):
        for j in range(K2):
            W.append(W1[:, int(i)] * W2[:, int(j)])

    return tf.transpose(tf.stack(W))
