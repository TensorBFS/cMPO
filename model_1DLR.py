""" Construction of cMPO for several models.

   T: cMPO. The structure of cMPO:
            --                              --
            | I + dtau Q  -- sqrt(dtau) R -- |
            |                                |
            |       |                        |
            | sqrt(dtau) L        P          |
            |       |                        |
            --                              --
    W: the unitary gate U that connects the cMPO to its transpose is given by
                --       --
                | 1 0 .. 0|
                | 0       |
          U =   | :   W   |
                | 0       |
                --       --
        this can save some time since <l| is related to |r> by U.
        set W = None if U doesn't exist or you don't want to find it
    ph_leg: the physical dimension of the cMPO
    d: the virtual dimension of the cMPO minus 1

"""

import torch
from pcmpo_1Dlongrange import pcmpo
import numpy as np

torch.manual_seed(42)

class spin_half(object):
    """ The Pauli matrices and S+, S-
    """
    def __init__(self, dtype, device):
        # Pauli matrices
        self.Id = torch.eye(2, dtype=dtype, device=device)
        self.X = torch.tensor([[0.0, 1.0],  [1.0, 0.0]], dtype=dtype, device=device)
        self.iY = torch.tensor([[0.0, 1.0],  [-1.0, 0.0]], dtype=dtype, device=device)
        self.Z = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=dtype, device=device)
        self.Sp = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=dtype, device=device)
        self.Sm = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=dtype, device=device) 

class ising(object):
    """ transverse field Ising model
        H = -J \sum_{<i,j>} Z_i Z_j - Gamma \sum_i X_i 
    """
    def __init__(self, Gamma, J, dtype, device):
        s = spin_half(dtype, device)
        Q = Gamma * s.X
        L = np.sqrt(J) * s.Z.view(1,2,2)
        R = np.sqrt(J) * s.Z.view(1,2,2)
        P = torch.zeros(1,1,2,2, dtype=dtype, device=device)
        self.T = pcmpo(Q, L, R, P) 
        self.W = torch.diag(torch.tensor([1], dtype=dtype, device=device))
        self.ph_leg = 2
        self.d = 1

class xxz_spm(object):
    """ XXZ model
        H = \sum_{<i,j>} (-Jxy Sx_i Sx_j - Jxy Sy_i Sy_j + Jz Sz_i Sz_j)
        The cMPO is implemented by S+, S-, and Sz
    """
    def __init__(self, Jz, Jxy, dtype, device):
        s = spin_half(dtype, device)
        Jz_sign = np.sign(Jz)
        Jz_abs = np.abs(Jz)

        Q = torch.zeros(2,2, dtype=dtype, device=device)
        L = torch.cat((
            s.Sp.view(1,2,2)*np.sqrt(Jxy/2),
            s.Sm.view(1,2,2)*np.sqrt(Jxy/2),
            np.sqrt(Jz_abs)*s.Z.view(1,2,2)/2
            ), dim=0)
        R = torch.cat((
            s.Sm.view(1,2,2)*np.sqrt(Jxy/2),
            s.Sp.view(1,2,2)*np.sqrt(Jxy/2),
            -Jz_sign * np.sqrt(Jz_abs)*s.Z.view(1,2,2)/2
            ), dim=0)
        P = torch.zeros(3,3,2,2, dtype=dtype, device=device)
        self.T = pcmpo(Q, L, R, P)
        self.W = torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, -Jz_sign]], dtype=dtype, device=device)
        self.ph_leg = 2
        self.d = 3

class xxz(object):
    """ XXZ model
        H = \sum_{<i,j>} (-Jxy Sx_i Sx_j - Jxy Sy_i Sy_j + Jz Sz_i Sz_j)
        Alternative construction of the cMPO with Sx, iSy, Sz
    """
    def __init__(self, Jz, Jxy, dtype, device):
        s = spin_half(dtype, device)
        Jz_sign = np.sign(Jz)
        Jz_abs = np.abs(Jz)

        Q = torch.zeros(2,2, dtype=dtype, device=device)
        L = torch.cat((
             np.sqrt(Jxy)/2 * s.X.view(1,2,2),
             np.sqrt(Jxy)/2 * s.iY.view(1,2,2),
             np.sqrt(Jz_abs)/2 * s.Z.view(1,2,2)
             ), dim=0 )
        R = torch.cat((
             np.sqrt(Jxy)/2 * s.X.view(1,2,2),
             -np.sqrt(Jxy)/2 * s.iY.view(1,2,2),
             -Jz_sign * np.sqrt(Jz_abs)/2 * s.Z.view(1,2,2)
             ), dim=0 )
        P = torch.zeros(3,3,2,2, dtype=dtype, device=device)
        self.T = pcmpo(Q, L, R, P)
        self.W = torch.diag(torch.tensor([1,-1,-Jz_sign], dtype=dtype, device=device))
        self.ph_leg = 2 
        self.d = 3

class ising_NNN(object):
    """ TFIM with next-nearest-neighboring interaction
        H = -J \sum_{<i,j>} Z_i Z_j -J2 \sum_{<<i,j>>} Z_i Z_j - Gamma \sum_i X_i 
    """
    def __init__(self, Gamma, J, J2, dtype, device):
        s = spin_half(dtype, device)

        Q = Gamma * s.X
        L = torch.cat((
             np.sqrt(J/2) * s.Z.view(1,2,2),
             np.sqrt(J/2) * s.Z.view(1,2,2),
             ), dim=0 )
        R = torch.cat((
             np.sqrt(J/2) * s.Z.view(1,2,2),
             np.sqrt(J/2) * s.Z.view(1,2,2),
             ), dim=0 )
        P0 = torch.tensor([[0, 2*J2/J], [0, 0]], dtype=dtype, device=device)
        P = torch.einsum('mn,ab->mnab', P0, s.Id) 
        self.T = pcmpo(Q, L, R, P) 
        self.W = torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=device)
        self.ph_leg =2
        self.d = 2

class ising_expLR(object):
    """ TFIM with exponentially decaying long-range interaction
        H = -J \sum_{i,j} \exp(-alpha * |i-j|) Z_i Z_j - Gamma \sum_i X_i 
    """
    def __init__(self, Gamma, J, alpha, dtype, device):
        s = spin_half(dtype, device)
        Q = Gamma * s.X
        L = np.sqrt(J) * np.exp(-alpha/2) * s.Z.view(1,2,2)
        R = np.sqrt(J) * np.exp(-alpha/2) * s.Z.view(1,2,2)
        P = np.exp(-alpha) * s.Id.view(1,1,2,2)
        self.T = pcmpo(Q, L, R, P) 
        self.W = torch.diag(torch.tensor([1], dtype=dtype, device=device))
        self.ph_leg =2
        self.d = 1

class ising_powLR(object):
    """ TFIM with power-law decaying long-range interaction
        H = -J \sum_{i,j} |j-i|^{-alpha} Z_i Z_j - Gamma \sum_i X_i 
        the power-law decaying interaction is approximated with a sum of exponentials
         |j-i|^{-alpha} = \sum_k^K mu_k exp(-l_k |j-i|)
        initial guess for mu, l , and choice of K is suggested by arXiv:physics/0605149
    """
    def __init__(self, Gamma, J, alpha, dtype, device):
        K = int(np.log(500)/np.log(3.87)) 
        eta = np.float_power(500, 1/K)
        mu0 = 1/np.sum( 
             [np.float_power(eta, -ix*alpha) * np.exp(-alpha/eta**ix) for ix in range(K)]
        )
        mu_vec0 = np.float_power(eta, -np.arange(K) * alpha) * mu0
        l_vec0 = alpha / np.float_power(eta, np.arange(K))

        mu_vec = torch.nn.Parameter(torch.tensor(mu_vec0, dtype=dtype, device=device))
        l_vec = torch.nn.Parameter(torch.tensor(l_vec0, dtype=dtype, device=device))

        # save parameters (borrow the class from cmps)
        from pcmpo_1Dlongrange import data_cmps, datasave, dataload 
        para_data = data_cmps(mu_vec, l_vec)
        path='power_fit_parameters_alpha{:.2f}.pt'.format(alpha)

        # optimization obtained over range of 200 sites
        fa = lambda x, alpha: 1/np.float_power(x, alpha)
        fb = lambda x, mu_vec, l_vec: mu_vec @ torch.exp(-l_vec * x)
        def func(mu_vec, l_vec):
            y = 0
            for j in np.arange(200)+1:
                y += (fa(j, alpha) - fb(j, mu_vec, l_vec))**2
            return y

        try: 
            dataload(para_data, path)
            print('parameters loaded successfully')
        except:
            print('parameters failed to load, fitting')
            optimizer0 = torch.optim.LBFGS([mu_vec, l_vec], max_iter = 20, tolerance_grad=0, tolerance_change=0, line_search_fn="strong_wolfe")
            def closure0():
                optimizer0.zero_grad()
                loss = func(mu_vec, l_vec)
                loss.backward()
                return loss

            counter = 0
            loss0 = 9.99e99
            while counter < 2:
                loss = optimizer0.step(closure0)
                print('--> ' + '{:.12f}'.format(loss.item()), end=' \r')
                if np.isclose(loss.item(), loss0, rtol=1e-9, atol=1e-9):
                    counter += 1
                loss0 = loss.item()
            datasave(para_data, path)

        mu_vec, l_vec = mu_vec.detach(), l_vec.detach()
        self.mu_vec = mu_vec
        self.l_vec = l_vec
        print('final loss', func(mu_vec, l_vec).item())
    
        s = spin_half(dtype, device)
        Q = Gamma * s.X
        mu_abs_vec = torch.abs(mu_vec)
        mu_sgn_vec = torch.sign(mu_vec)
        L = torch.einsum('m,ab->mab', torch.exp(-l_vec/2)*torch.sqrt(J*mu_abs_vec), s.Z)
        R = torch.einsum('m,ab->mab', torch.exp(-l_vec/2)*torch.sqrt(J*mu_abs_vec)*mu_sgn_vec, s.Z)
        P = torch.einsum('m,mn,ab->mnab',torch.exp(-l_vec), torch.eye(K, dtype=dtype, device=device), s.Id)
        self.T = pcmpo(Q, L, R, P) 
        self.W = torch.diag(mu_sgn_vec)
        self.ph_leg = 2
        self.d = K

