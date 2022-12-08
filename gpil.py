import torch
from torch.nn import Parameter

import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
import pyro.ops.stats as stats

import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy

import nfoursid.nfoursid
import nfoursid.kalman

import pandas as pd
import numpy as np
from itertools import product


def GPIL(*, Y, N_pseudo: int, M_dim: int):
    gp_f, gp_g, loc_0, cov_0 = GPIL_init(Y=Y, N_pseudo=N_pseudo, M_dim=M_dim)
    gp_f, gp_g, loc_0, cov_0 = GPIL_EM(
        Y=Y, gp_f=gp_f, gp_g=gp_g, E_X=loc_0, cov_X=cov_0)
    return gp_f, gp_g


def GPIL_init(Y, N_pseudo: int, M_dim: int):
    Y_filt, lv_state, lv_cov = init_latent_state(Y=Y, M_dim=M_dim)

    loc_0 = torch.zeros((N_pseudo,))  # not sure what these are for yet..
    cov_0 = torch.eye(N_pseudo)

    gp_f = init_spg(X_0=lv_state[:-1], y_0=lv_state[1:],
                    N_pseudo=N_pseudo, M_dim=M_dim)
    gp_g = init_spg(X_0=lv_state, y=Y_filt, N_pseudo=N_pseudo, M_dim=M_dim)

    return gp_f, gp_g, loc_0, cov_0


def GPIL_EM(*, Y, gp_f, gp_g, E_X, cov_X, N_EM_steps=10):
    for k in range(N_EM_steps):
        moments = GP_ADF(Y=Y, gp_f=gp_f, gp_g=gp_g, E_X=E_X, cov_X=cov_X)
        gp_f, gp_g = amax_likelihood(
            moments=moments, Y=Y, gp_f=gp_f, gp_g=gp_g)

    return gp_f, gp_g


def GP_ADF(*, Y, gp_f, gp_g, E_0, cov_0):

    E_x, cov_X = E_0, cov_0

    E_y_list = []
    cov_y_list = []
    for t, y_t in enumerate(Y):
        # inducing points
        e, v = gp_g.Xu
        E_y, cov_y, cov_Xy = GPUR(
            gp_list=gp_g, loc_in=E_X, cov_in=cov_X, X=e, y=v)
        E_X, cov_X = filter_update(
            y=y_t, E_X=E_X, cov_X=cov_X, E_y=E_y, cov_y=cov_y, cov_Xy=cov_Xy)

        a, b = gp_f.Xu
        E_x, cov_X, _ = GPUR(gp_list=gp_f, loc_in=E_X, cov_in=cov_X, X=a, y=b)

    return E_y_list, cov_y_list


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

        kernel = gp.kernels.RBF(
            input_dim=M_dim, lengthscale=torch.ones(M_dim))

        ### FIT INITIAL KERNEL ###
        gp_0 = gp.models.GPRegression(X=X_0, y=y_0, kernel=kernel)
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

    params = [gp.parameters() for gp in gp_f+gp_g]

    optimizer = torch.optim.Adam(
        params=params, lr=0.005)

    raise NotImplemented
    for k in range(num_steps):
        optimizer.zero_grad()
        loss = marginal_loglikelihood(
            moments=moments, Y=Y, gp_f=gp_f, gp_g=gp_g)
        loss.backward()
        optimizer.step()


def marginal_loglikelihood(*, moments, Y, gp_f, gp_g):
    # this seems to be the right probability dist but it is not a function of the parameters

    return torch.sum(dist.MultivariateNormal(loc=E_y, covariance_matrix=cov_y).log_prob(Y_t) for Y_t, (E_y, cov_y) in zip(Y, moments))


def GPUR(*, X, y, loc_in, cov_in, gp_list):
    '''Gaussian Process Regression with Uncertain inputs'''

    dim = len(gp_list)

    # Page 52, thesis
    beta_list = []
    q = torch.zeros((dim,))

    L_list = []
    inv_L_list = []

    ker_var_list = []

    var_noise = torch.zeros((dim,))

    for a, gp_a in enumerate(gp_list):
        with torch.nograd():
            Ka = gp_a.kernel(X)
        beta_a = torch.linalg.solve(Ka, y[a])
        beta_list.append(beta_a)

        ker_var, ard_length = list(gp_a.kernel.parameters())

        L = torch.diag(ard_length)
        inv_L = torch.diag(torch.tensor(1/l for l in ard_length))

        L_list.append(L)
        inv_L.append(inv_L)

        ker_var_list.append(ker_var)

        arg_v = X - loc_in
        q[a] = ker_var * \
            torch.det(cov_in @ inv_L + torch.eye()) * \
            rbf_point(M=(cov_in+L), v=arg_v)  # TODO check scaling

    M = torch.zeros((len(gp_list), len(gp_list)))
    for a, b in mrange((len(gp_list),)*2):
        beta_a, Q_ab = None, None  # here to prevent misuse of old value
        for i, j in mrange((len(X),)*2):

            den = L_list[a] + L_list[b]
            z_ij = L_list[b] @ torch.linalg.solve(
                den, X[i]) + L_list[a] @ torch.linalg.solve(den, X[j])

            arg = inv_L_list[a]+inv_L_list[b]
            invArg = torch.diag(1/el for el in torch.diagonal(arg))

            Q_ab = ker_var_list[a]*ker_var_list[b] * \
                torch.pow(torch.det((inv_L_list[a] + inv_L_list[b])@cov_in +
                                    torch.eye(cov_in.shape)), -0.5)*rbf_point(M=(L_list[a]+L_list[b]), v=(X[i]-X[j]))*rbf_point(M=(invArg+cov_in), v=(z_ij-loc_in))
        beta_a = beta_list[a]
        M[a][b] = beta_a.T*Q_ab*beta_a

    loc_star = torch.zeros((dim,))
    for e in range(dim):
        loc_star[e] = q[e]*torch.sum(beta_list[a])
    cov_star = M - torch.dot(loc_star, loc_star.T) + \
        torch.diag(var_noise)  # Might not work

    return loc_star, cov_star


def filter_update(*, y, E_X, cov_X, E_y, cov_y, cov_Xy):
    innovation = y - E_y
    # gain = cov_Xy @ torch.inverse(cov_y)

    # could use more stable versions of these computations
    ret_E_X = E_X + cov_Xy @ torch.linalg(cov_y, innovation)
    ret_cov_X = cov_X - cov_Xy @ torch.linalg(cov_y.T, cov_Xy.T)
    # return filtered Expected X, and Cov X
    return ret_E_X, ret_cov_X


### HELPER UTILITIES ###
def mrange(*sizes):
    return product(*[range(size) for size in sizes])


def rbf_point(*, M, v):
    '''returns exp(-1/2vT*invM*v)'''
    return torch.exp(-1/2*v.T@(torch.linalg.solve(M, v)))
