import shelve
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
import nfoursid.state_space

import pandas as pd
import numpy as np
from itertools import product


def GPIL(*, Y, N_pseudo: int, M_dim: int):
    gp_f, gp_g, loc_0, cov_0, ss_cov = GPIL_init(
        Y=Y, N_pseudo=N_pseudo, M_dim=M_dim)
    gp_f, gp_g, loc_0, cov_0 = GPIL_EM(
        Y=Y, gp_f=gp_f, gp_g=gp_g, E_0=loc_0, cov_0=cov_0)
    return gp_f, gp_g, ss_cov


def GPIL_init(Y, N_pseudo: int, M_dim: int):
    Y_filt, lv_state, lv_cov, ss_cov = init_latent_state(Y=Y, M_dim=M_dim)

    lv_state = lv_state.t()

    loc_0 = torch.zeros((N_pseudo,))  # not sure what these are for yet..
    # I am expecting these test inputs should be filtered / aligned with the inducing points of gp_f
    cov_0 = torch.eye(N_pseudo)

    gp_f = init_spg(X_0=lv_state[:-1, :], y_0=lv_state[1:, :].t(),
                    N_pseudo=N_pseudo, M_dim=M_dim)
    gp_g = init_spg(X_0=lv_state, y_0=Y_filt, N_pseudo=N_pseudo, M_dim=M_dim)

    return gp_f, gp_g, loc_0, cov_0, ss_cov


def GPIL_EM(*, Y, gp_f, gp_g, E_0, cov_0, N_EM_steps=10):
    for k in range(N_EM_steps):
        # moments = GP_ADF(Y=Y, gp_f=gp_f, gp_g=gp_g, E_X=E_0, cov_X=cov_0)
        gp_f, gp_g = amax_likelihood(
            E_0=E_0, cov_0=cov_0, Y=Y, gp_f=gp_f, gp_g=gp_g)

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

    ss_cov = (ss, cov)

    return *kalf(Y=Y, ss=ss, cov=cov), ss_cov


def kalf(Y, ss, cov):
    '''Kalman filter given a State Space model'''

    kalman = nfoursid.kalman.Kalman(ss, cov)
    for y in Y:
        kalman.step(np.asarray([[y]]), u=np.asarray([[]]).reshape((0, 1)))

    lv_hat = t2d_from_n4sid(kalman.x_filtereds)
    cov_lv_hat = t2d_from_n4sid(kalman.p_filtereds)
    Y_filt = t2d_from_n4sid(kalman.y_filtereds)
    return Y_filt, lv_hat, cov_lv_hat


def t2d_from_n4sid(res):
    '''returns a 2D torch.Tensor from nfoursid interface'''
    return torch.tensor(np.concatenate(res, axis=1))


def init_spg(*, X_0: torch.tensor, y_0: torch.tensor, N_pseudo: int, M_dim: int) -> tuple:

    # num_outputs = len(y_0)

    N_points, num_outputs = y_0.shape

    gp_list = []
    kernel_list = []

    for k, y in enumerate(y_0):

        kernel = gp.kernels.RBF(
            input_dim=M_dim)

        ### FIT INITIAL KERNEL ###
        gp_0 = gp.models.GPRegression(
            X=X_0, y=y, kernel=kernel, noise=torch.tensor(1e-6))
        gp.util.train(gpmodule=gp_0, num_steps=300)

        ### SAMPLE INDUCING POINTS ###
        indices = [int(len(X_0)*rv)
                   for rv in dist.Uniform(0, 1).sample((N_pseudo,))]
        Xu = X_0[indices]

        ### FIT SPARSE GP USING OPTIMAL KERNEL ###
        sgp = gp.models.GPLVM(
            gp.models.SparseGPRegression(X=X_0, y=y_0, Xu=Xu, kernel=kernel, noise=torch.tensor(1e-6)))

        ### TUNE INDUCING POINTS AND KERNEL ###
        gp.util.train(gpmodule=sgp, num_steps=300)

        gp_list.append(sgp)
        kernel_list.append(kernel)

    return gp_list, kernel_list


def amax_likelihood(*, E_0, cov_0, Y, gp_f, gp_g, num_steps: int):

    params = [gp.parameters() for gp in gp_f+gp_g]

    optimizer = torch.optim.Adam(
        params=params, lr=0.005)

    for k in range(num_steps):
        optimizer.zero_grad()
        loss = marginal_loglikelihood(
            E_0=E_0, cov_0=cov_0, Y=Y, gp_f=gp_f, gp_g=gp_g)
        loss.backward()
        optimizer.step()


def marginal_loglikelihood(*, E_0, cov_0, Y, gp_f, gp_g):

    moments = GP_ADF(Y=Y, gp_f=gp_f, gp_g=gp_g, E_0=E_0, cov_0=cov_0)
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
    V = torch.zeros((len(X), dim))
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
            rbf_point(M=(cov_in+L), v=arg_v)

        sum_vec = torch.zeros((len(X),))
        for i in range(len(X)):
            sum_vec += beta_a[i]*q[a][i]*(X[i] - loc_in)
        V[:][a] = cov_in @ torch.linalg.solve(cov_in + L, sum_vec)

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
        M[a][b] = beta_a.t()*Q_ab*beta_a

    loc_star = torch.zeros((dim,))
    for e in range(dim):
        loc_star[e] = q[e]*torch.sum(beta_list[a])
    cov_star = M - torch.matmul(loc_star, loc_star.t()) + \
        torch.diag(var_noise)

    return loc_star, cov_star, V


def filter_update(*, y, E_X, cov_X, E_y, cov_y, cov_Xy):
    innovation = y - E_y
    # gain = cov_Xy @ torch.inverse(cov_y)

    # could use more stable versions of these computations
    ret_E_X = E_X + cov_Xy @ torch.linalg(cov_y, innovation)
    ret_cov_X = cov_X - cov_Xy @ torch.linalg(cov_y.t(), cov_Xy.t())
    # return filtered Expected X, and Cov X
    return ret_E_X, ret_cov_X


### HELPER UTILITIES ###
def mrange(*sizes):
    return product(*[range(size) for size in sizes])


def rbf_point(*, M, v):
    '''returns exp(-1/2vT*invM*v)'''
    return torch.exp(-1/2*v.t()@(torch.linalg.solve(M, v)))


class gpts():
    '''
    Usage: 
    1. gpts.train() offline on a series
    2. gpts.init_predict() on a series
    3. gpts.predict_next() to retrieve the expected result
    '''

    def __init__(self, Y_train=None, **kwargs) -> None:
        self.params = None
        if Y_train is not None:
            self.train(Y=Y_train, **kwargs)

        self.y = None
        self.y_cov = None
        self.x_loc = None
        self.x_cov = None

    def train(self, *, Y, N_pseudo: int, M_dim: int):
        gp_f, gp_g, ss_cov = GPIL(Y=Y, N_pseudo=N_pseudo, M_dim=M_dim)
        self.params = gpts.get_params(gp_f=gp_f, gp_g=gp_g, kf_fit=ss_cov)

    def init_predict(self, *, Y):
        print('len(Y) should be sufficient to burn in HMM')
        # broad steps:
        # 1. initialise latent state using the kalman filter
        # 2. run inference, using fixed parameters

        Y_filt, X_est, X_cov = kalf(
            Y=Y, ss=self.params.ss, cov=self.params.cov)

        self.y = Y_filt
        self.x_loc = X_est
        self.x_cov = X_cov

    def predict_N(self, N_steps, u_k=None):
        if u_k is not None:
            assert N_steps == len(u_k)
        
        

    def get_params(*, gp_f, gp_g, ss_cov):
        return gpts_params(gp_f=gp_f, gp_g=gp_g, ss_cov=ss_cov)


class gpts_params():
    def __init__(self, gp_f, gp_g, ss_cov) -> None:
        # extract params from the sparse gaussian regressions
        
        # f_X = gp_f.Xu
        # f_y = gp_f.

        # self.params = {
        #     'f': {
        #         'X': f_X,
        #         'y': f_y,
        #         'cov': f_var,
        #     },
        #     'g': {

        #     },
        #     'ss': {
        #         'lti': ss_cov[0]
        #         'cov': ss_cov[1]
        #     }
        # }
        pass


def dump_scope(f='shelfout'):
    fn = f'tmp/{f}'
    with shelve.open(fn) as shelf:
        for key in dir():
            try:
                shelf[key] = locals()[key]
            except TypeError:
                pass


def load_scope(f='shelfout'):
    fn = f'tmp/{f}'
    with shelve.open(fn) as shelf:
        for key in shelf:
            locals()[key] = shelf[key]


def test_env():
    ss = nfoursid.state_space(
        a=np.array([[0, 1], [-1, -0.5]]),
        b=np.array([[0], [1]]),
        c=np.array([[1, 0]]),
        d=np.array([[0]]),
    )


if __name__ == '__main__':

    # df = pd.read_feather('dp_df.feather')
    # Y = torch.tensor(df['o2pp'][:1000])
    # ts = gpts(Y_train=Y, N_pseudo=50, M_dim=4)
    # ts.predict(Y)

    init_spg(X_0=None, y_0=None, N_pseudo=None, M_dim=None)
