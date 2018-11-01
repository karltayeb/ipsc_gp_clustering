import numpy as np
import gpflow
import os
import sys
import pickle
import copy
import math
from gpflow import params_as_tensors_for
from Assignments import Assignments


class Logger(gpflow.actions.Action):
    def __init__(self, model, assignments):
        self.model = model
        self.assignments = assignments
        self.logf = []
        self.q_mu = [np.zeros_like(model.q_mu.value)]
        # self.assignments_list = [copy.deepcopy(assignments)]
        self.assignments_list = []
        self.likelihood_params_list = [model.likelihood.read_values()]
        self.kernel_params_list = [model.kern.read_values()]

    def run(self, ctx):
        if (ctx.iteration % 10) == 0:
            # self.assignments_list.append(copy.deepcopy(self.assignments))
            self.likelihood_params_list.append(
                copy.deepcopy(self.model.likelihood.read_values()))
            self.kernel_params_list.append(
                copy.deepcopy(self.model.kern.read_values()))

        if (ctx.iteration % 1) == 0:
            # update to be correct lower bound w/ assignment probs/entropytras
            # likelihood = ctx.session.run(self.model.likelihood_tensor)
            likelihood = full_lml(self.model, 1000000)
            likelihood += self.assignments.likelihood()
            entropy = self.assignments.entropy()

            self.logf.append(entropy - likelihood)

            if np.allclose(self.model.q_mu.value - self.q_mu[-1], 0):
                sys.exit()
            else:
                print(np.sum((self.model.q_mu.value - self.q_mu[-1]) ** 2))
                self.q_mu.append(copy.deepcopy(self.model.q_mu.value))


class Saver(gpflow.actions.Action):
    def __init__(self, model, assignments, logger, save_path='./model'):
        self.model = model
        self.assignments = assignments
        self.logger = logger
        self.save_path = save_path

        save_dir = '/'.join(save_path.split('/')[:-1])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def run(self, ctx):
        print('SAVING: ', self.save_path)

        for param in self.model.parameters:
            param.trainable = True
        self.model.weights.trainable = False

        model_params = self.model.read_trainables()
        with open(self.save_path, 'wb') as f:
            pickle.dump([model_params, self.assignments,
                         self.logger.logf, self.logger.likelihood_params_list,
                         self.logger.kernel_params_list,
                         self.logger.assignments_list], f)


class UpdateAssignments(gpflow.actions.Action):
    def __init__(self, model, mean_model, assignments, x, y):
        self.model = model
        self.assignments = assignments
        self.x = x
        self.y = y

    def run(self, ctx):
        print('UPDATING ASSIGNMENTS')
        self.assignments.update_assignments(
            self.model,
            self.x.transpose(1, 2, 0), self.y.transpose(1, 2, 0))

        weights = self.assignments.compute_weights()
        self.model.weights = weights


class UpdateMeans(gpflow.actions.Action):
    def __init__(self, model, mean_model):
        self.model = model
        self.mean_model = mean_model

    def run(self, ctx):
        print('UPDATING TRAJECTORIES')
        # turn on trainables
        X = self.model.X.value
        Y = self.model.Y.value
        weight_idx = self.model.weight_idx.value
        weights = self.model.weights.value

        # compress observations from main model
        x_agg, time_index = np.unique(X, return_index=True)
        wy_sums = np.array(
            [np.sum(wy, axis=0) for wy in
             np.split(weights[weight_idx] * Y, time_index[1:])])
        w_agg = np.array(
            [np.sum(w, axis=0) for w in
             np.split(weights[weight_idx], time_index[1:])]) + 1e-8
        y_agg = wy_sums / w_agg

        # set trainabels
        for param in self.model.parameters:
            param.trainable = False

        self.mean_model.q_mu.trainable = True
        self.mean_model.q_sqrt.trainable = True

        # put new data in mean model
        self.mean_model.X = x_agg[:, None]
        self.mean_model.Y = y_agg
        self.mean_model.weights = w_agg

        # optimize with new data passed in
        opt = gpflow.train.ScipyOptimizer()
        opt.minimize(self.mean_model)
        # evaluate(lambda **kwargs: opt.minimize(m_bar, **kwargs), feed_dict)


class SetMeans(gpflow.actions.Action):
    def __init__(self, model, mean_model):
        self.model = model
        self.mean_model = mean_model

    def run(self, ctx):
        print('UPDATING TRAJECTORIES')
        # turn on trainables
        X = self.model.X.value
        Y = self.model.Y.value
        weight_idx = self.model.weight_idx.value
        weights = self.model.weights.value

        # compress observations from main model
        x_agg, time_index = np.unique(X, return_index=True)
        wy_sums = np.array(
            [np.sum(wy, axis=0) for wy in
             np.split(weights[weight_idx] * Y, time_index[1:])])
        w_agg = np.array(
            [np.sum(w, axis=0) for w in
             np.split(weights[weight_idx], time_index[1:])])
        y_agg = wy_sums / w_agg

        # set trainabels
        for param in self.model.parameters:
            param.trainable = False

        self.mean_model.q_mu.trainable = True
        self.mean_model.q_sqrt.trainable = True

        # put new data in mean model
        self.mean_model.X = x_agg[:, None]
        self.mean_model.Y = y_agg
        self.mean_model.weights = w_agg


class UpdateHyperparameters(gpflow.actions.Action):
    def __init__(self, model):
        self.model = model

    def run(self, ctx):
        print("UPDATING HYPERPARAMETERS")
        # turn on trainables
        for param in self.model.parameters:
            param.trainable = False

        for param in self.model.kern.parameters:
            param.trainable = True

        for param in self.model.likelihood.parameters:
            param.trainable = True

        opt = gpflow.train.AdamOptimizer()
        opt.minimize(self.model)


class OLDAssignments:
    def __init__(self, pi, psi, rho, Phi, Lambda, Gamma):
        self.pi = pi
        self.psi = psi
        if rho is None:
            rho = np.array([1.0, 0.0]).astype(np.float64)
        self.rho = rho

        self.Phi = Phi
        self.Lambda = Lambda

        if Gamma is None:
            Gamma = np.ones((Lambda.shape[1], 2))
            Gamma[:, 1] = 0
        self.Gamma = Gamma


def make_feed(param, value):
    return {param.unconstrained_tensor: param.transform.backward(value)}


def full_lml(model, batch_size):
    with params_as_tensors_for(model):
        tf_x, tf_y, tf_w = model.X, model.Y, model.weight_idx

    lml = 0.0
    num_batches = int(math.ceil(len(model.X._value) / batch_size))
    for mb in range(num_batches):
        start = mb * batch_size
        finish = (mb + 1) * batch_size
        x_mb = model.X._value[start:finish, :]
        y_mb = model.Y._value[start:finish, :]
        w_mb = model.weight_idx._value[start:finish]
        mb_lml = model.compute_log_likelihood(
            feed_dict={tf_x: x_mb, tf_y: y_mb, tf_w: w_mb})
        lml += mb_lml * len(x_mb)

    lml = lml / model.X._value.size
    return lml


def evaluate(func, param_feed_dict):
    tensor_feed_dict = {}
    for param, value in param_feed_dict.items():
        tensor_feed_dict.update(make_feed(param, value))
    return func(feed_dict=tensor_feed_dict)


def multinomial_entropy(p):
    return -1 * np.nansum(p * np.log(p))


def train_mixsvgp(model, mean_model, assignments, x, y, iterations, save_path):
    logger = Logger(model, assignments)
    assignment_update = UpdateAssignments(
        model, mean_model, assignments, x, y)
    mean_update = UpdateMeans(model, mean_model)

    hyperparam_update = UpdateHyperparameters(model)
    saver = Saver(model, assignments, logger, save_path)

    actions = [mean_update, hyperparam_update,
               assignment_update, logger, saver]

    gpflow.actions.Loop(actions, stop=iterations)()
    model.anchor(model.enquire_session())
    return logger
