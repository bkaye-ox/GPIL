import numpy as np
import torch

import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
import pyro.ops.stats as stats

# import gpy


def GPIL(Y, N: int, M: int):
    LDS = n4sid(Y, M)
    mean_0 = np.zeros((M,))
    cov_0 = np.eye(M)

    gpil = GPIL_Params()
    mean_0, cov_0, gpil = EM_max(hp=gpil, mean_0=mean_0, cov_0=cov_0)
    return mean_0, cov_0, gpil


def EM_max(*, Y, hp, mean_0, cov_0, iters=10):

    # some how optimise:
    mean_X, cov_X = mean_0, cov_0
    for k in range(iters):
        # E step:
        pred_Y, pred_cov_Y, mean_X, cov_X = GP_ADF(
            Y=Y, mean_0=mean_X, cov_0=cov_X, hp=hp)

        # M-step
        hp = argmax_hp(hp_0=hp)

    return mean_X, cov_X


def loss_log_likelihood(*, Y, gp):
    loss =


def argmax_hp(*, gp: gp.GPModel):
    raise NotImplemented

    # use (4.26) ie argmax(l(theta))
    # l(p) = sum_t ( log p(y_t|y_(1:t-1), theta) )

    gp.util.train(gpmodule=gp, num_steps=100)

    ### using gradient descent / pytorch ###

    hp_star = hp_0
    return hp_star


def log_likelihood(*, Y, gp: gp.GPModel):
    torch.sum(dist.Normal(gp.model()).log_prob(Y))


class GPIL_Params():
    def __init__(self, N: int) -> None:
        self.f = GP_Hparam(N)
        self.g = GP_Hparam(N)
    # def __getitem__(self, key):
    #     if key == 'g':
    #         return self.f
    #     if key == 'f':
    #         return self.g
    #     raise KeyError


class GP_Hparam():
    def __init__(self, N: int) -> None:
        pass
        self.theta = 0
        self.x = [0 for k in range(N)]
        self.y = [0 for k in range(N)]


def GP_ADF(*, Y, mean_0, cov_0, hp: GPIL_Params):
    '''Gaussian Process Assumed Density Filter'''
    mean_X, cov_X = mean_0, cov_0

    pred_y = []
    pred_y_var = []
    for t, Y_t in enumerate(Y):
        mean_Y, cov_Y, cov_XY = GPUR(hp.g, mean_X, cov_X)  # t|1:t-1
        mean_X, cov_X = filter_update(
            Y_t, mean_X, cov_X, mean_Y, cov_Y, cov_XY)  # t|1:t
        mean_X, cov_X = GPUR(hp.f, mean_X, cov_X)  # t+1|1:t

        pred_y.append(mean_Y)
        pred_y_var.append(cov_Y)

    return pred_y, pred_y_var, mean_X, cov_X


def SGP(mean_l):
    '''Stochastic Gaussian Process'''
    raise NotImplemented
    return GP_Hparam()


def filter_update():
    return None, None


def GPUR():
    return None, None


def n4sid():
    A = None
    return A
