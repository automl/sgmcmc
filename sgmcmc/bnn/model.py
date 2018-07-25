
import time
from collections import deque
import logging

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import lasagne

from ..theano_mcmc import SGHMCSampler
from ..utils import sharedX, floatX, shuffle
from .priors import WeightPrior, LogVariancePrior



# normalization for the BNN (copied from RoBO)
def zero_mean_unit_var_normalization(X, mean=None, std=None):
    if mean is None:
        mean = np.mean(X, axis=0)
    if std is None:
        std = np.std(X, axis=0)

    X_normalized = (X - mean) / std

    return X_normalized, mean, std


def zero_mean_unit_var_unnormalization(X_normalized, mean, std):
    return X_normalized * std + mean



class HMCBNN(object):

    def __init__(self, f_net_fun,
                 burn_in=2000, capture_every=50, log_every=100,
                 update_prior_every=100, out_type='Gaussian',
                 updater=None, weight_prior=WeightPrior(),
                 variance_prior=LogVariancePrior(1e-4, 0.01),
                 n_target_nets=100, rng=None):
        if rng:
            self._srng = rng
        else:
            self._srng = RandomStreams(np.random.randint(1, 2147462579))
        self.updater = SGHMCSampler(precondition=True, rng=rng)
        self.f_net_fun = f_net_fun
        self.f_net = f_net_fun()
        if n_target_nets > 1:
            self.f_nets = [f_net_fun() for i in range(n_target_nets)]
        else:
            self.f_nets = []
        
        self.out_type = out_type
        self.weight_prior = weight_prior
        self.variance_prior = variance_prior
        self.burn_in = burn_in
        self.capture_every = capture_every
        self.update_prior_every = update_prior_every
        self.steps = 0
        self.bsize = 32
        self.log_every = log_every
        self.n_output_dim = len(self.f_net.output_shape)
        self.prepared = False
        if self.n_output_dim not in [2, 3]:
            raise ValueError('HMCBNN expects either 2 or 3 dimensional output from the net')
        Xbatch = T.matrix()
        self.out_fun = theano.function([Xbatch], lasagne.layers.get_output(self.f_net, Xbatch, deterministic=True))
        self.mcmc_samples = []
        if n_target_nets > 1:
            m_t, v_t = self.approximate_mean_and_var(Xbatch)
            self.predict_approximate_fun = theano.function([Xbatch], [m_t, v_t])

    def _log_like(self, X, Y, n_examples):
        f_out = lasagne.layers.get_output(self.f_net, X)
        f_mean = f_out[:, 0].reshape((-1, 1))
        f_log_var = f_out[:, 1].reshape((-1, 1))
        f_var_inv = 1. / (T.exp(f_log_var) + 1e-8)
        MSE = T.square(Y - f_mean)
        if self.out_type == 'Gaussian':
            log_like = T.sum(T.sum(-MSE * (0.5*f_var_inv) - 0.5*f_log_var, axis=1))
        else:
            raise RuntimeError('{} not implemented'.format(self.out_type))
        # scale by batch size to make this work nicely with the updaters above
        log_like /= T.cast(X.shape[0], theano.config.floatX)
        #priors, scale these by dataset size for the same reason
        # prior for the variance
        self.tn_examples = sharedX(np.float32(n_examples))
        log_like += self.variance_prior.log_like(f_log_var, n_examples) / self.tn_examples
        # prior for the weights
        log_like += self.weight_prior.log_like(lasagne.layers.get_all_params(self.f_net, regularizable=True)) / self.tn_examples
        return log_like, T.sum(MSE)


    def prepare_for_train(self, shape, bsize, epsilon, **kwargs):
        n_examples = shape[0]
        self.n_examples = n_examples
        self.steps = 0
        self.mcmc_samples = []
        self.params = lasagne.layers.get_all_params(self.f_net, trainable=True)
        self.variance_prior.prepare_for_train(n_examples)
        wdecay = self.weight_prior.prepare_for_train(self.params, n_examples)
        # setup variables for training
        Xbatch = T.matrix()
        Ybatch = T.matrix()
        print("... preparing costs")
        log_like, mse = self._log_like(Xbatch, Ybatch, n_examples)
        self.costs = -log_like
        print("... preparing updates")
        updates, burn_in_updates = self.updater.prepare_updates(self.costs, self.params, epsilon,
                                                                scale_grad=n_examples, **kwargs)
        # handle batch normalization (which we however don't use anyway)
        bn_updates = [u for l in lasagne.layers.get_all_layers(self.f_net) for u in getattr(l,'bn_updates',[])]
        updates += bn_updates

        # we have two functions, one for the burn in phase, including the burn_in_updates and one during sampling
        self.compute_cost_burn_in = theano.function([Xbatch, Ybatch], (self.costs, mse), updates=updates + burn_in_updates)
        self.compute_cost = theano.function([Xbatch, Ybatch], (self.costs, mse), updates=updates)

        # Data dependent initialization if required
        init_updates = [u for l in lasagne.layers.get_all_layers(self.f_net) for u in getattr(l,'init_updates',[])]
        
        self.data_based_init = theano.function([Xbatch], lasagne.layers.get_output(self.f_net, Xbatch, init=True), updates=init_updates)
        self.prepared = True
        self.first_step = True


    def update_for_train(self, shape, bsize, epsilon, retrain=True, **kwargs):
        n_examples = shape[0]
        self.n_examples = n_examples
        self.tn_examples.set_value(np.float32(n_examples))
        # reset the network parameters without having to recompile the theano graph
        if retrain:
            new_net = self.f_net_fun()
            new_params = lasagne.layers.get_all_param_values(new_net)
            lasagne.layers.set_all_param_values(self.f_net, new_params)
        self.first_step = True
        self.steps = 0
        self.mcmc_samples = []
        self.weight_prior.update_for_train(n_examples)
        self.variance_prior.update_for_train(n_examples)
        self.updater.reset(n_examples, epsilon, reset_opt_params=retrain, **kwargs)
    
    def step(self, X, Y, capture=True):
        if self.steps <= self.burn_in:
            cost, mse = self.compute_cost_burn_in(X, Y)
        else:
            cost, mse = self.compute_cost(X, Y)
        if capture and (self.steps > self.burn_in) \
           and (self.capture_every > 0) and (self.steps % self.capture_every == 0):
            self.mcmc_samples.append(lasagne.layers.get_all_param_values(self.f_net))
            if len(self.f_nets) > 0:
                # replace one of the f_nets
                idx = (len(self.mcmc_samples) - 1) % len(self.f_nets)
                #idx = np.random.randint(len(self.f_nets))
                #idx_mcmc = np.random.randint(len(self.mcmc_samples))
                lasagne.layers.set_all_param_values(self.f_nets[idx], self.mcmc_samples[-1])
        if self.steps % self.log_every == 0:
            print("Step: {} stored_samples : {} WD : {},  NLL = {}, MSE = {}, Noise = {}".format(self.steps, len(self.mcmc_samples), self.weight_prior.get_decay().get_value(), cost, mse, float(np.exp(self.f_net.b.get_value()))))
        if self.steps > 1 and self.steps % self.update_prior_every == 0:
            self.weight_prior.update(lasagne.layers.get_all_params(self.f_net, regularizable=True))
        self.steps += 1
        return cost

    def approximate_mean_and_var(self, Xbatch):
        if len(self.f_nets) == 0:
            raise RuntimeError("You called approximate_mean_and_var but n_target_nets is <= 1")
        mean_y = None
        ys2var = None
        mean_pred = None
        # use law of total variance to compute the overall variance
        for net in self.f_nets:
            f_out = lasagne.layers.get_output(net, Xbatch, deterministic=True)
            y = f_out[:, 0:1]
            var = T.exp(f_out[:, 1:2]) + 1e-16
            if mean_y is None:
                mean_y = y
                ys2var = T.square(y) + var
            else:
                mean_y += y
                ys2var += T.square(y) + var
        n_nets = T.cast(len(self.f_nets), theano.config.floatX)
        mean_y /= n_nets
        ys2var /= n_nets
        total_var = ys2var - T.square(mean_y)
        return mean_y, total_var

    def predict_approximate(self, X):
        return self.predict_approximate_fun(floatX(X))
    
    def predict(self, X):
        ys, var = self.sample_predictions(floatX(X))
        # compute predictive mean
        mean_pred = np.mean(ys, axis=0)
        # use the law of total variance to compute the overall variance
        var_pred = np.mean(ys ** 2 + var, axis=0) - mean_pred ** 2
        return mean_pred, var_pred

    def sample_predictions(self, X):
        y = []
        var = []
        for sample in self.mcmc_samples:
            lasagne.layers.set_all_param_values(self.f_net, sample)
            f_out = self.out_fun(X)
            y.append(f_out[:, 0])
            var.append(np.exp(f_out[:, 1]) + 1e-16)
                
        return np.asarray(y), np.asarray(var)
        

    def predict_online(self, Xtest, n_samples, X, Y, capture_every=50):
        # this runs the markov chain forward until we have made enough predictions
        old_cap = self.capture_every
        self.capture_every = capture_every
        n_steps = int(n_samples * self.capture_every)
        indices = np.random.permutation(np.arange(len(X)))
        y = []
        var = []
        for s in range(n_steps):
            # sample a random batch
            start = int(np.random.randint(len(indices - self.bsize)))
            idx = indices[start:start+int(self.bsize)]
            xmb = X[idx]
            ymb = Y[idx]
            # and push the markov chain forward
            self.step(xmb, ymb, capture=False)
            # predict if we are in a capture step
            if s % self.capture_every == 0:
                f_out = self.out_fun(Xtest)
                y.append(f_out[:, 0])
                #var.append(np.log(1. + np.exp(f_out[:, 1])))
                var.append(np.exp(f_out[:, 1]))
        self.capture_every = old_cap
        return np.asarray(y), np.asarray(var)
    
    def train(self, X, Y, n_steps, retrain=True, bsize=32, epsilon=1e-2, **kwargs):
        self.X = X
        self.Y = Y        
        
        if n_steps < self.burn_in:
            raise ValueError("n_steps must be larger than burn_in")
        print('X shape : {}'.format(X.shape))
        print('Y shape : {} '.format(Y.shape))
        ndata = X.shape[0]
        self.bsize = bsize
        if X.shape[0] < 2*self.bsize:
            self.bsize = X.shape[0]
        n_batches = int(np.ceil(ndata / self.bsize))
        n_epochs = int(np.floor(n_steps / n_batches))
        #data_per_batch = ndata / self.bsize
        if not self.prepared:
            self.prepare_for_train(X.shape, self.bsize, epsilon, **kwargs)
        else:
            self.update_for_train(X.shape, self.bsize, epsilon, retrain=retrain, **kwargs)
        for e in range(n_epochs):
            X, Y = shuffle(X, Y)
            #print("Starting epoch: {}".format(e))
            batches = len(X) // self.bsize
            for b in range(batches):
                start = b*self.bsize
                xmb = X[start:start+self.bsize]
                ymb = Y[start:start+self.bsize]
                if self.first_step:
                    print("Performing data based initialization")
                    self.data_based_init(xmb)
                    self.first_step = False
                self.step(xmb, ymb)



class BayesianNeuralNetwork(object):

    def __init__(self,
                 get_net, 
                 sampling_method="sghmc",
                 n_nets=100, l_rate=1e-3,
                 mdecay=5e-2, n_iters=5 * 10**4,
                 bsize=20, burn_in=1000,
                 sample_steps=100,
                 precondition=True, normalize_output=True,
                 normalize_input=True, rng=None):
        """
        Bayesian Neural Networks use Bayesian methods to estimate the posterior distribution of a neural
        network's weights. This allows to also predict uncertainties for test points and thus makes
        Bayesian Neural Networks suitable for Bayesian optimization.

        This module uses stochastic gradient MCMC methods to sample from the posterior distribution together See [1]
        for more details.

        [1] J. T. Springenberg, A. Klein, S. Falkner, F. Hutter
            Bayesian Optimization with Robust Bayesian Neural Networks.
            In Advances in Neural Information Processing Systems 29 (2016).

        Parameters
        ----------
        get_net : func
            function that returns a network specification.

        sampling_method : str
            Determines the MCMC strategy:
            "sghmc" = Stochastic Gradient Hamiltonian Monte Carlo
            "sgld" = Stochastic Gradient Langevin Dynamics

        n_nets : int
            The number of samples (weights) that are drawn from the posterior

        l_rate : float
            The step size parameter for SGHMC

        mdecay : float
            Decaying term for the momentum in SGHMC

        n_iters : int
            Number of MCMC sampling steps without burn in

        bsize : int
            Batch size to form a mini batch

        burn_in : int
            Number of burn-in steps before the actual MCMC sampling begins

        precondition : bool
            Turns on / off preconditioning. See [1] for more details

        normalize_input : bool
            Turns on / off zero mean unit variance normalization of the input data

        normalize_output : bool
            Turns on / off zero mean unit variance normalization of the output data

        rng : np.random.RandomState()
            Random number generator
        """

        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng

        lasagne.random.set_rng(self.rng)

        self.sampling_method = sampling_method
        self.n_nets = n_nets
        self.l_rate = l_rate
        self.mdecay = mdecay
        self.n_iters = n_iters
        self.bsize = bsize
        self.burn_in = burn_in
        self.precondition = precondition
        self.is_trained = False
        self.normalize_output = normalize_output
        self.normalize_input = normalize_input
        self.get_net = get_net

        self.sample_steps = sample_steps
        self.samples = deque(maxlen=n_nets)

        self.variance_prior = LogVariancePrior(1e-6, 0.01)
        self.weight_prior = WeightPrior(alpha=1., beta=1.)

        self.Xt = T.matrix()
        self.Yt = T.matrix()

        self.X = None
        self.x_mean = None
        self.x_std = None
        self.y = None
        self.y_mean = None
        self.y_std = None


    def train(self, X, y, *args, **kwargs):
        """
        Trains the model on the provided data.

        Parameters
        ----------
        X: np.ndarray (N, D)
            Input data points. The dimensionality of X is (N, D),
            with N as the number of points and D is the number of features.
        y: np.ndarray (N,)
            The corresponding target values.

        """

        # Clear old samples
        start_time = time.time()

        self.net = self.get_net(n_inputs=X.shape[1])

        nll, mse = self.negative_log_likelihood(self.net, self.Xt, self.Yt, X.shape[0], self.weight_prior, self.variance_prior)
        params = lasagne.layers.get_all_params(self.net, trainable=True)

        seed = self.rng.randint(1, 100000)
        srng = theano.sandbox.rng_mrg.MRG_RandomStreams(seed)

        if self.sampling_method == "sghmc":
            self.sampler = SGHMCSampler(rng=srng, precondition=self.precondition, ignore_burn_in=False)
        elif self.sampling_method == "sgld":
            self.sampler = SGLDSampler(rng=srng, precondition=self.precondition)
        else:
            logging.error("Sampling Strategy % does not exist!" % self.sampling_method)

        self.compute_err = theano.function([self.Xt, self.Yt], [mse, nll])
        self.single_predict = theano.function([self.Xt], lasagne.layers.get_output(self.net, self.Xt))

        self.samples.clear()

        if self.normalize_input:
            self.X, self.x_mean, self.x_std = zero_mean_unit_var_normalization(X)
        else:
            self.X = X

        if self.normalize_output:
            self.y, self.y_mean, self.y_std = zero_mean_unit_var_normalization(y)
        else:
            self.y = y

        self.sampler.prepare_updates(nll, params, self.l_rate, mdecay=self.mdecay,
                                     inputs=[self.Xt, self.Yt], scale_grad=X.shape[0])

        logging.info("Starting sampling")

        # Check if we have enough data points to form a minibatch
        # otherwise set the batchsize equal to the number of input points
        if self.X.shape[0] < self.bsize:
            self.bsize = self.X.shape[0]
            logging.error("Not enough datapoint to form a minibatch. "
                          "Set the batchsize to {}".format(self.bsize))

        i = 0
        while i < self.n_iters and len(self.samples) < self.n_nets:
            if self.X.shape[0] == self.bsize:
                start = 0
            else:
                start = np.random.randint(0, self.X.shape[0] - self.bsize)

            xmb = floatX(self.X[start:start + self.bsize])
            ymb = floatX(self.y[start:start + self.bsize, None])

            if i < self.burn_in:
                _, nll_value = self.sampler.step_burn_in(xmb, ymb)
            else:
                _, nll_value = self.sampler.step(xmb, ymb)

            if i % 512 == 0 and i <= self.burn_in:
                total_err, total_nll = self.compute_err(floatX(self.X), floatX(self.y).reshape(-1, 1))
                t = time.time() - start_time
                logging.info("Iter {:8d} : NLL = {:11.4e} MSE = {:.4e} "
                             "Time = {:5.2f}".format(i, float(total_nll),
                             float(total_err), t))

            if i % self.sample_steps == 0 and i >= self.burn_in:
                total_err, total_nll = self.compute_err(floatX(self.X), floatX(self.y).reshape(-1, 1))
                t = time.time() - start_time
                self.samples.append(lasagne.layers.get_all_param_values(self.net))
                logging.info("Iter {:8d} : NLL = {:11.4e} MSE = {:.4e} "
                             "Samples= {} Time = {:5.2f}".format(i,
                                                                      float(total_nll),
                                                                      float(total_err),
                                                                      len(self.samples), t))
            i += 1
        self.is_trained = True

    def negative_log_likelihood(self, f_net, X, y, n_examples, weight_prior, variance_prior):

        f_out = lasagne.layers.get_output(f_net, X)
        f_mean = f_out[:, 0].reshape((-1, 1))

        f_log_var = f_out[:, 1].reshape((-1, 1))

        f_var_inv = 1. / (T.exp(f_log_var) + 1e-16)
        mse = T.square(y - f_mean)
        log_like = T.sum(T.sum(-mse * (0.5 * f_var_inv) - 0.5 * f_log_var, axis=1))
        # scale by batch size to make this work nicely with the updaters above
        log_like /= T.cast(X.shape[0], theano.config.floatX)
        # scale the priors by the dataset size for the same reason
        # prior for the variance
        tn_examples = T.cast(n_examples, theano.config.floatX)
        log_like += variance_prior.log_like(f_log_var) / tn_examples
        # prior for the weights
        params = lasagne.layers.get_all_params(f_net, trainable=True)
        log_like += weight_prior.log_like(params) / tn_examples

        return -log_like, T.mean(mse)


    def predict(self, X_test, return_individual_predictions=False, *args, **kwargs):
        """
        Returns the predictive mean and variance of the objective function at
        the given test points.

        Parameters
        ----------
        X_test: np.ndarray (N, D)
            Input test points

        return_individual_predictions: bool
            If set to true than the individual predictions of all samples are returned.

        Returns
        ----------
        np.array(N,)
            predictive mean
        np.array(N,)
            predictive variance

        """

        if not self.is_trained:
            logging.error("Model is not trained!")
            return

        # Normalize input
        if self.normalize_input:
            X_, _, _ = zero_mean_unit_var_normalization(X_test, self.x_mean, self.x_std)
        else:
            X_ = X_test

        f_out = []
        theta_noise = []
        for sample in self.samples:
            lasagne.layers.set_all_param_values(self.net, sample)
            out = self.single_predict(X_)
            f_out.append(out[:, 0])
            theta_noise.append(np.exp(out[:, 1]))

        f_out = np.asarray(f_out)
        theta_noise = np.asarray(theta_noise)

        if return_individual_predictions:
            if self.normalize_output:
                f_out = zero_mean_unit_var_unnormalization(f_out, self.y_mean, self.y_std)
                theta_noise *= self.y_std**2
            return f_out, theta_noise

        m = np.mean(f_out, axis=0)
        # Total variance
        # v = np.mean(f_out ** 2 + theta_noise, axis=0) - m ** 2
        v = np.mean((f_out - m) ** 2, axis=0)

        if self.normalize_output:
            m = zero_mean_unit_var_unnormalization(m, self.y_mean, self.y_std)
            v *= self.y_std ** 2

        return m, v
