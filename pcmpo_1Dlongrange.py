""" utility functions for cMPO method
"""

import numpy as np
import torch 
import os, io, subprocess
import os.path
os.environ['OMP_NUM_THREADS']='15'
torch.set_num_threads(15)
torch.manual_seed(42)

from eigh import EigenSolver
eigensolver = EigenSolver.apply

def eigensolver(M):
    """ Eigensolver
        manually symmetrize M before the eigen decomposition
    """
    return EigenSolver.apply(0.5*(M+M.t()))

def log_trace_expm(beta, mat):
    """ calculates log(tr(exp(beta * mat)) )
    """
    w, _ = eigensolver(mat)
    return torch.logsumexp(beta*w, dim=0)

class pcmpo(object):
    """ the object for cMPO
        dim: the physical dimension of the cMPO
        the structure of cMPO 
            --                              --
            | I + dtau Q  -- sqrt(dtau) R -- |
            |                                |
            |       |                        |
            | sqrt(dtau) L        P          |
            |       |                        |
            --                              --
    """
    def __init__(self, Q, L, R, P):
        self.dim = Q.shape[0]
        self.dtype = Q.dtype
        self.device = Q.device
        self.Q = Q # 2 leg: D x D
        self.L = L # 3 leg: d x D x D
        self.R = R # 3 leg: d x D x D
        self.P = P # 4 leg: d x d x D x D

    def detach(self):
        """ return the detached cMPO object, clear autograd information
        """
        return pcmpo(self.Q.detach(), self.L.detach(), self.R.detach(), self.P.detach())

    def project(self, U):
        """ perform a unitary transformation in the imaginary-time direction
            if U is a square matrix, this is a guage transformation
        """
        Q = U.t() @ self.Q @ U
        L = U.t() @ self.L @ U
        R = U.t() @ self.R @ U
        P = U.t() @ self.P @ U
        return pcmpo(Q, L, R, P)

    def t(self):
        """ give the transpose of the cMPO
        """
        Q = self.Q
        L = self.R
        R = self.L
        P = torch.einsum('abmn->bamn', self.P)
        return pcmpo(Q, L, R, P)

class pcmps(object):
    """ the object for cMPS
        dim: the physical dimension of the cMPS
        the structure of cMPS 
            --            --
            | I + dtau Q   |
            |              |
            |       |      |
            | sqrt(dtau) R |
            |       |      |
            --            --
    """
    def __init__(self, Q, R):
        self.dim = Q.shape[0]
        self.dtype = Q.dtype
        self.device = Q.device
        self.Q = Q
        self.R = R

    def detach(self):
        """ return the detached cMPS object, clear autograd information
        """
        return pcmps(self.Q.detach(), self.R.detach())

    def project(self, U):
        """ perform a unitary transformation in the imaginary-time direction
            if U is a square matrix, this is a guage transformation
        """
        Q = U.t() @ self.Q @ U
        R = U.t() @ self.R @ U
        return pcmps(Q, R)

    def diagQ(self):
        """ transform the cMPS to the gauge where Q is a diagonalized matrix 
        """
        _, U = eigensolver(self.Q)
        return self.project(U)

def multiply(W, mps):
    """ multiply a matrix to the left of the cMPS
             --        --   --            --
             | 1 0 ... 0|   | I + dtau Q   |
             | 0        |   |              |
             | :        |   |       |      |
             | :    W   |   | sqrt(dtau) R |
             | 0        |   |       |      |
             --        --   --            --
    """
    dtype, device = mps.dtype, mps.device
    R1 = torch.einsum('mn, nab->mab', W, mps.R)
    return pcmps(mps.Q, R1)

def act(mpo, mps):
    """ act the cmps to the right of cmpo
             --                              --   --            --
             | I + dtau Q  -- sqrt(dtau) R -- |   | I + dtau Q   |
             |                                |   |              |
             |       |                        |   |       |      |
             | sqrt(dtau) L        P          |   | sqrt(dtau) R |
             |       |                        |   |       |      |
             --                              --   --            --
    """
    dtype, device = mps.dtype, mps.device
    Do, Ds = mpo.dim, mps.dim
    d = mps.R.shape[0]
    Io = torch.eye(Do, dtype=dtype, device=device) 
    Is = torch.eye(Ds, dtype=dtype, device=device)

    Q_rslt = torch.einsum('ab,cd->acbd', mpo.Q, Is).contiguous().view(Do*Ds, Do*Ds) \
           + torch.einsum('ab,cd->acbd', Io, mps.Q).contiguous().view(Do*Ds, Do*Ds) \
           + torch.einsum('mab,mcd->acbd', mpo.R, mps.R).contiguous().view(Do*Ds, Do*Ds) 
    R_rslt = torch.einsum('mab,mcd->macbd', mpo.L, Is.repeat(d,1,1)).contiguous().view(d, Do*Ds, Do*Ds) \
           + torch.einsum('mnab,ncd->macbd', mpo.P, mps.R).contiguous().view(d, Do*Ds, Do*Ds)

    return pcmps(Q_rslt, R_rslt)

def Lact(mps, mpo):
    """ act the cmps to the left of cmpo
          --            --  --                              --   
          | I + dtau Q   |  | I + dtau Q  -- sqrt(dtau) R -- |   
          |              |  |                                |   
          |       |      |  |       |                        |   
          | sqrt(dtau) R |  | sqrt(dtau) L        P          |   
          |       |      |  |       |                        |   
          --            --  --                              --   
    """
    dtype, device = mps.dtype, mps.device
    Do, Ds = mpo.dim, mps.dim
    d = mps.R.shape[0]

    Tmps = act(mpo.t(), mps)
    Q = torch.einsum('abcd->badc', Tmps.Q.view(Do, Ds, Do, Ds)).contiguous().view(Do*Ds, Do*Ds)
    R = torch.einsum('mabcd->mbadc', Tmps.R.view(d, Do, Ds, Do, Ds)).contiguous().view(d, Do*Ds, Do*Ds)
    return pcmps(Q, R)

def density_matrix(mps1, mps2):
    """ construct the K matrix corresponding to <mps1|mps2>
       --                          --  --             --     
       |                            |  | I + dtau Q2   |     
       |                            |  |       |       |     
       | I + dtau Q1 sqrt(dtau) R1  |  | sqrt(dtau) R2 |   = I + dtau K 
       |                            |  |       |       |     
       --                          --  --             --     
    """
    dtype, device= mps1.dtype, mps1.device
    D1, D2 = mps1.dim, mps2.dim
    I1 = torch.eye(mps1.dim, dtype=dtype, device=device)
    I2 = torch.eye(mps2.dim, dtype=dtype, device=device) 

    M = torch.einsum('ab,cd->acbd', mps1.Q, I2).contiguous().view(D1*D2, D1*D2) \
      + torch.einsum('ab,cd->acbd', I1, mps2.Q).contiguous().view(D1*D2, D1*D2) \
      + torch.einsum('mab,mcd->acbd', mps1.R, mps2.R).contiguous().view(D1*D2, D1*D2) 
    return M

def ln_ovlp(mps1, mps2, beta):
    """ calculate log(<mps1|mps2>)
    """
    M = density_matrix(mps1, mps2)
    return log_trace_expm(beta, M)
def Fidelity(psi, mps, beta):
    """ calculate log [ <psi|mps> / sqrt(<psi|psi>) ]
    """
    up = ln_ovlp(psi, mps, beta)
    dn = ln_ovlp(psi, psi, beta)
    return up - 0.5*dn

def rdm_update(mps1, mps2, beta, chi):
    """ initialize the isometry 
        keep the chi largest weights of the reduced density matrix
        return the isometry
    """
    rho = density_matrix(mps1, mps2)
    D1, D2 = mps1.dim, mps2.dim
    rdm = torch.einsum('xaxb->ab', rho.view(D1, D2, D1, D2))
    w, v = eigensolver(rdm)
    P = v[:, -chi:]
    return P

def mera_update(mps, beta, chi, tol=1e-10, alpha=0.5, maxiter=20):
    """ update the isometry with iterative SVD update
        mps: the original cMPS
        beta: inverse temperature
        chi: target bond dimension
        return the compressed cMPS
    """
    P = rdm_update(mps, mps, beta, chi)
    last = 9.9e9
    step = 0
    while step < maxiter:
        mps_new = mps.project(P.requires_grad_())
        loss = ln_ovlp(mps, mps_new, beta)
        diff = abs(loss.item() - last)

        if (diff < tol): break

        grad = torch.autograd.grad(loss, P)[0]
        last = loss.item() 
        step += 1
    
        U, _, V = torch.svd(grad)
        #https://mathoverflow.net/questions/262560/natural-ways-of-interpolating-unitary-matrices
        #https://groups.google.com/forum/#!topic/manopttoolbox/2zhx67doXaU
        #interpolate between unitary matrices
        mix = alpha * U@V.t() + (1.-alpha) * P.data
        #then retraction back to unitary
        U, _, V = torch.svd(mix)
        P = U@V.t()
    
    return mps_new 

def variational_compr(mps, beta, chi, chkp_loc, tol=1e-8):
    """ variationally optimize the compressed cMPS 
        mps: the original cMPS
        beta: the inverse temperature
        chi: target bond dimension
        chkp_loc: the location to save check point datafile
        tol: tolerance
        return the compressed cMPS
    """
    psi = mera_update(mps, beta, chi)
    psi = psi.diagQ()

    Q = torch.nn.Parameter(torch.diag(psi.Q))
    R = torch.nn.Parameter(psi.R)
    psi_data = data_cmps(Q, R)

    optimizer = torch.optim.LBFGS([Q, R], max_iter=20, tolerance_grad=0, tolerance_change=0, line_search_fn="strong_wolfe") 

    def closure():
        optimizer.zero_grad()
        psi = pcmps(torch.diag(Q), R)
        loss = - Fidelity(psi, mps, beta)
        loss.backward()
        return loss

    counter = 0
    loss0 = 9.99e99
    while counter < 2:
        loss = optimizer.step(closure)
        print('--> ' + '{:.12f}'.format(loss.item()), end='\r')
        if np.isclose(loss.item(), loss0, rtol=tol, atol=tol):
            counter += 1
        loss0 = loss.item()

    # "normalize"
    with torch.no_grad():
        Q -= torch.max(Q)
        psi = pcmps(torch.diag(Q), R) 
    # checkpoint
    datasave(psi_data, chkp_loc)

    return psi.detach()

# utility functions for save and load datafile
class data_cmps(torch.nn.Module):
    def __init__(self, Q, R):
        super(data_cmps, self).__init__()
        self.Q = Q 
        self.R = R
def datasave(model, path):
    torch.save(model.state_dict(), path)
def dataload(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()

