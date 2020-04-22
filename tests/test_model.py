import torch
import numpy as np
import functools
import sys, os
testdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(testdir+"/..")
from cmpo import *
from model import *

tnp = lambda a, b, c: torch.einsum('ab,cd,ef->acebdf', a, b, c).view(a.shape[0]*b.shape[0]*c.shape[0], a.shape[1]*b.shape[1]*c.shape[1])

def decorator_test_model(func):
    @functools.wraps(func)
    def wrapper_test_model(*args, **kwargs):
        model, H = func(*args, **kwargs)
        check_close = lambda a, b: np.allclose(a, b, atol=1e-4)
        T = model.T

        # the unitary matrix W connects the cMPO to its spatial transpose
        # check the correctness of W
        if not model.W is None:
            assert check_close(torch.einsum('mn,nab->mab', model.W, T.L), T.R)
            assert check_close(torch.einsum('mn,nab->mab', model.W, T.R), T.L)
            assert check_close(torch.einsum('pm,mnab,nq->pqab', model.W, T.P, model.W), torch.einsum('mnab->nmab', T.P))

        # a three-site open-boundary system can be viewed as <Lpsi|T|Rpsi>
        # check that -K = H
        # this tests the correctness of the cMPO construction
        Rpsi = cmps(T.Q, T.L)
        Lpsi = cmps(T.Q, T.R)
        TR = act(T, Rpsi)
        K1 = density_matrix(Lpsi, TR)
        LT = Lact(Lpsi, T)
        K2 = density_matrix(LT, Rpsi)
        if not check_close(-K1, H):
            import pdb; pdb.set_trace()
            print(K1+H)
        assert check_close(K1, K2)
        assert check_close(-K1, H)

    return wrapper_test_model

@decorator_test_model
def test_ising():
    Gamma, J = 0.78, 0.87
    def func(Gamma, J):
        # cmpo
        model = ising(Gamma=Gamma, J=J, dtype=torch.float64, device='cpu')
        # manually construct a 3-site Hamiltonian for test
        s = spin_half(dtype=torch.float64, device='cpu')
        H = -J * (tnp(s.Z, s.Z, s.Id) + tnp(s.Id, s.Z, s.Z)) - Gamma * (tnp(s.X, s.Id, s.Id) + tnp(s.Id, s.X, s.Id) + tnp(s.Id, s.Id, s.X))
        return model, H 
    return func(Gamma, J)

@decorator_test_model
def test_xxz_spm():
    Jz, Jxy = 0.78, 0.87
    def func(Jz, Jxy):
        # cmpo
        model = xxz_spm(Jz=Jz, Jxy=Jxy, dtype=torch.float64, device='cpu')
        # manually construct a 3-site Hamiltonian for test
        s = spin_half(dtype=torch.float64, device='cpu')
        H = -Jxy * (tnp(s.X/2, s.X/2, s.Id) + tnp(s.Id, s.X/2, s.X/2)) \
            +Jxy * (tnp(s.iY/2, s.iY/2, s.Id) + tnp(s.Id, s.iY/2, s.iY/2)) \
            +Jz  * (tnp(s.Z/2, s.Z/2, s.Id) + tnp(s.Id, s.Z/2, s.Z/2))
        return model, H 
    return func(Jz, Jxy)

@decorator_test_model
def test_xxz():
    Jz, Jxy = 0.78, 0.87
    def func(Jz, Jxy):
        # cmpo
        model = xxz(Jz=Jz, Jxy=Jxy, dtype=torch.float64, device='cpu')
        # manually construct a 3-site Hamiltonian for test
        s = spin_half(dtype=torch.float64, device='cpu')
        H = -Jxy * (tnp(s.X/2, s.X/2, s.Id) + tnp(s.Id, s.X/2, s.X/2)) \
            +Jxy * (tnp(s.iY/2, s.iY/2, s.Id) + tnp(s.Id, s.iY/2, s.iY/2)) \
            +Jz  * (tnp(s.Z/2, s.Z/2, s.Id) + tnp(s.Id, s.Z/2, s.Z/2))
        return model, H 
    return func(Jz, Jxy)

@decorator_test_model
def test_isingNNN():
    Gamma, J, J2 = 0.678, 0.786, 0.867
    def func(Gamma, J, J2):
        # cmpo
        model = ising_NNN(Gamma=Gamma, J=J, J2=J2, dtype=torch.float64, device='cpu')
        # manually construct a 3-site Hamiltonian for test
        s = spin_half(dtype=torch.float64, device='cpu')
        H = -J * (tnp(s.Z, s.Z, s.Id) + tnp(s.Id, s.Z, s.Z)) \
            -J2 * tnp(s.Z, s.Id, s.Z) \
            -Gamma * (tnp(s.X, s.Id, s.Id) + tnp(s.Id, s.X, s.Id) + tnp(s.Id, s.Id, s.X))
        return model, H 
    return func(Gamma, J, J2)

@decorator_test_model
def test_ising_expLR():
    Gamma, J, alpha = 0.678, 0.786, 0.867
    def func(Gamma, J, alpha):
        # cmpo
        model = ising_expLR(Gamma=Gamma, J=J, alpha=alpha, dtype=torch.float64, device='cpu')
        # manually construct a 3-site Hamiltonian for test
        s = spin_half(dtype=torch.float64, device='cpu')
        H = -J*np.exp(-alpha) * (tnp(s.Z, s.Z, s.Id) + tnp(s.Id, s.Z, s.Z)) \
            -J*np.exp(-alpha*2) * tnp(s.Z, s.Id, s.Z) \
            -Gamma * (tnp(s.X, s.Id, s.Id) + tnp(s.Id, s.X, s.Id) + tnp(s.Id, s.Id, s.X))
        return model, H 
    return func(Gamma, J, alpha)

@decorator_test_model
def test_ising_powLR():
    Gamma, J, alpha = 0.678, 0.786, 1.867 # alpha should not be too small
    def func(Gamma, J, alpha):
        # cmpo
        model = ising_powLR(Gamma=Gamma, J=J, alpha=alpha, dtype=torch.float64, device='cpu')
        # manually construct a 3-site Hamiltonian for test
        s = spin_half(dtype=torch.float64, device='cpu')
        H = -J*np.float_power(1, -alpha) * (tnp(s.Z, s.Z, s.Id) + tnp(s.Id, s.Z, s.Z)) \
            -J*np.float_power(2, -alpha) * tnp(s.Z, s.Id, s.Z) \
            -Gamma * (tnp(s.X, s.Id, s.Id) + tnp(s.Id, s.X, s.Id) + tnp(s.Id, s.Id, s.X))
        return model, H 
    return func(Gamma, J, alpha)

#@decorator_test_model
#def test_ising_powLR():
#    return ising_powLR(Gamma=1, J=1, alpha=2, dtype=torch.float64, device='cpu') 
