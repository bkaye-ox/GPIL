import numpy as np
import scipy.signal as sps

import nfoursid.nfoursid
import nfoursid.kalman
import nfoursid.state_space

import pandas as pd

from torch import tensor

import plotly.express as px

class SSID():
    def __init__(self, Y, u=None, lv_dim=4) -> None:
        pass
        self.Y = np.array(Y)
        self.u = np.array(u)
        self.M = lv_dim

    def train(self, nblocks=10):

        df = pd.DataFrame()

        out_cols, in_cols = [], []
        for k,y in enumerate(self.Y):
            df[f'y{k}'] = y
            out_cols.append(f'y{k}')

        if self.u is not None:
            # df['u'] = self.u
            for k,u in enumerate(self.u):
                df[f'u{k}'] = u
                in_cols.append(f'u{k}')


        

        n4sid = nfoursid.nfoursid.NFourSID(
            df,
            output_columns=out_cols,
            input_columns=in_cols,
            num_block_rows=nblocks
        )

        n4sid.subspace_identification()
        ss, cov = n4sid.system_identification(rank=self.M)
        kalman = nfoursid.kalman.Kalman(ss, cov)
        for y, u in zip(self.Y.T, self.u.T):
            
            kalman.step(np.array([y]), u=np.array([u]).reshape((-1, 1)))

        lv_hat = __to_ndarray(kalman.x_filtereds)
        cov_lv_hat = __to_ndarray(kalman.p_filtereds)
        Y_filt = __to_ndarray(kalman.y_filtereds)
        Y_sd = __to_ndarray(kalman._measurement_and_state_standard_deviation(kalman.p_filtereds))
    
        return Y_filt, Y_sd, lv_hat, cov_lv_hat
        # return 


def __to_ndarray(d):
    return np.concatenate(d, axis=1)


def __to_tensor(d):
    return tensor(__to_ndarray(d))


class SS():
    def __init__(self, A, B, C, D=None, Ts=0.05) -> None:
        if B is None:
            B = np.array([[]])

            self.u_n = 0
        else:
            self.u_n = B.shape[1]

        self.x_n = A.shape[0]

        if D is None:
            D = np.array([[0] for k in range(self.u_n)])

        sys = sps.StateSpace(A, B, C, D).to_discrete(Ts)
        self.LTId = nfoursid.state_space.StateSpace(
            sys.A,
            sys.B,
            sys.C,
            sys.D,
        )

    def sim(self, N_steps, u=None, noise=0.1):
        if self.u_n > 0 and u is None:
            raise Exception('provide input or use SS.step')
        elif self.u_n > 0 and u.shape[1] != N_steps:
            raise Exception('input length mismatch')
        elif self.u_n > 0 and u.shape[0] != self.u_n:
            raise Exception('input dimension mismatch')

        gen = np.random.default_rng()
        for k in range(N_steps):

            n_out, _ = self.LTId.c.shape
            u_k = np.array([[1]])
            e_k = gen.normal(loc=0, scale=noise, size=(n_out, 1))

            self.LTId.step(e=e_k, u=u_k)

        return np.concatenate(self.LTId.ys[-N_steps:], axis=1).reshape(-1,)

    def __repr__(self) -> str:
        return f'A:{self.A}\nB:{self.B}\nC:{self.C}\nD:{self.D}'


if __name__ == '__main__':

    args = dict(a=np.array([[0, 1], [-1, -0.5]]),  # x1 = y(t) x2 = v(t)
                b=np.array([[0], [1]]),
                c=np.array([[1, 0]]),
                d=np.array([[0]]),)

    test_sys = SS(*args.values())

    sim_N = 100

    u_k = np.array([[1]*sim_N])
    Y_k = test_sys.sim(N_steps=sim_N, u=u_k)

    test_kalman = SSID(Y_k, u_k)
    Y_filt, _, _ = test_kalman.train()

    px.line(x=[k for k in range(sim_N)],y=[y for y in Y_filt])
    input('')