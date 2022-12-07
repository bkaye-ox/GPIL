import torch
from torch.nn import Parameter

import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
import pyro.ops.stats as stats

import nfoursid.nfoursid
import nfoursid.kalman

import pandas as pd
import numpy as np


def GPIL(*, Y, N_pseudo: int, M_dim: int):
    gp_f, gp_g = GPIL_init(Y=Y, N_pseudo=N_pseudo, M_dim=M_dim)
    gp_f, gp_g = GPIL_EM(Y=Y, gp_f=gp_f, gp_g=gp_g)
    return gp_f, gp_g


def GPIL_init(Y, N_pseudo: int, M_dim: int):
    Y_filt, lv_state, lv_cov = init_latent_state(Y=Y, M_dim=M_dim)

    loc_0 = torch.zeros((N_pseudo,))  # not sure what these are for yet..
    cov_0 = torch.eye(N_pseudo)

    gp_f = init_spg(X_0=lv_state[:-1], y_0=lv_state[1:],
                    N_pseudo=N_pseudo, M_dim=M_dim)
    gp_g = init_spg(X_0=lv_state, y=Y_filt, N_pseudo=N_pseudo, M_dim=M_dim)

    return gp_f, gp_g


def GPIL_EM(*, Y, gp_f, gp_g, N_EM_steps=10):
    for k in range(N_EM_steps):
        moments = GP_ADF(Y=Y, gp_f=gp_f, gp_g=gp_g)
        gp_f, gp_g = amax_likelihood(
            moments=moments, Y=Y, gp_f=gp_f, gp_g=gp_g)

    return gp_f, gp_g


def GP_ADF(*, Y, gp_f, gp_g):
    raise NotImplemented
    pass


def init_latent_state(*, Y: torch.tensor, M_dim: int) -> tuple:

    df = pd.DataFrame()
    df['y'] = Y

    n4sid = nfoursid.nfoursid.NFourSID(
        df,
        output_columns=['y'],
        num_block_rows=10
    )

    n4sid.subspace_identification()
    ss, cov = n4sid.system_identification(rank=M_dim)
    kalman = nfoursid.kalman.Kalman(ss, cov)
    for y in Y:
        kalman.step(np.asarray([[y]]), u=np.asarray([[]]).reshape((0, 1)))

    lv_hat = torch.tensor(kalman.x_filtereds)
    cov_lv_hat = torch.tensor(kalman.p_filtereds)
    Y_filt = torch.tensor(kalman.y_filtereds)

    return Y_filt, lv_hat, cov_lv_hat


def init_spg(*, X_0: torch.tensor, y_0: torch.tensor, N_pseudo: int, M_dim: int) -> tuple(list, list):

    num_outputs = len(y_0)

    gp_list = []
    kernel_list = []

    for k in range(num_outputs):

        kernel = gp_0.kernels.RBF(
            input_dim=M_dim, lengthscale=torch.ones(M_dim))

        ### FIT INITIAL KERNEL ###
        gp_0 = gp_0.models.GPRegression(X=X_0, y=y_0, kernel=kernel)
        gp_0.util.train(gpmodule=gp_0, num_steps=300)

        ### SAMPLE INDUCING POINTS ###
        indices = [int(len(X_0)*rv)
                   for rv in dist.Uniform(0, 1).sample((N_pseudo,))]
        Xu = X_0[indices]

        ### FIT SPARSE GP USING OPTIMAL KERNEL ###
        sgp = gp_0.models.GPLVM(
            gp_0.models.SparseGPRegression(X=X_0, y=y_0, Xu=Xu, kernel=kernel))

        ### TUNE INDUCING POINTS AND KERNEL ###
        gp.util.train(gpmodule=sgp, num_steps=300)

        gp_list.append(sgp)
        kernel_list.append(kernel)

    return gp_list, kernel_list


def amax_likelihood(*, moments, Y, gp_f, gp_g, num_steps: int):
    loss = marginal_loglikelihood()

    params = [gp.parameters() for gp in gp_f+gp_g]

    optimizer = torch.optim.Adam(
        params=params, lr=0.005)
    for k in range(num_steps):
        optimizer.zero_grad()
        loss = marginal_loglikelihood()
        loss.backward()
        optimizer.step()


def marginal_loglikelihood(*, moments, Y, gp_f, gp_g):
    pass
