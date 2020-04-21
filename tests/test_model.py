import torch
import numpy as np
from pcmpo_1Dlongrange import *
from model_1DLR import *
import functools

def decorator_test_model(func):
    @functools.wraps(func)
    def wrapper_test_model(*args, **kwargs):
        dtype=torch.float64
        device='cpu'

        model = func(*args, **kwargs)
        check_close = lambda a, b: np.allclose(a.numpy(), b.numpy())
        T = model.T

        # the unitary matrix W connects the cMPO to its spatial transpose
        # check the correctness of W
        if not model.W is None:
            assert check_close(torch.einsum('mn,nab->mab', model.W, T.L), T.R)
            assert check_close(torch.einsum('mn,nab->mab', model.W, T.R), T.L)
            assert check_close(torch.einsum('pm,mnab,nq->pqab', model.W, T.P, model.W), torch.einsum('mnab->nmab', T.P))

        # consistency check, two ways to calculate K matrix corresponding to <Lpsi|T|Rpsi>
        # check the results are the same
        Q = torch.rand(6,6, dtype=dtype, device=device)
        R = torch.rand(T.R.shape[0], 6, 6, dtype=dtype, device=device)
        Rpsi = pcmps(Q, R)
        Q = torch.rand(6,6, dtype=dtype, device=device)
        R = torch.rand(T.R.shape[0], 6, 6, dtype=dtype, device=device)
        Lpsi = pcmps(Q, R)
        TR = act(T, Rpsi)
        K1 = density_matrix(Lpsi, TR)
        LT = Lact(Lpsi, T)
        K2 = density_matrix(LT, Rpsi)
        assert check_close(K1, K2)

    return wrapper_test_model

@decorator_test_model
def test_ising():
    return ising(1.0, 1.0, dtype=torch.float64, device='cpu') 
@decorator_test_model
def test_xxz_spm():
    return xxz_spm(Jz=1, Jxy=1, dtype=torch.float64, device='cpu') 
@decorator_test_model
def test_xxz():
    return xxz(Jz=1, Jxy=1, dtype=torch.float64, device='cpu') 
@decorator_test_model
def test_ising_NNN():
    return ising_NNN(Gamma=1, J=1, J2=1, dtype=torch.float64, device='cpu') 
@decorator_test_model
def test_ising_expLR():
    return ising_expLR(Gamma=1, J=1, alpha=2, dtype=torch.float64, device='cpu') 
@decorator_test_model
def test_ising_powLR():
    return ising_powLR(Gamma=1, J=1, alpha=2, dtype=torch.float64, device='cpu') 
