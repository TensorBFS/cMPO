import torch
import numpy as np
from pcmpo_1Dlongrange import *

def test_project():
    """ Given a cMPS, check that the cMPS is unchanged after projecting a 
        unitary gate U without altering its bond dimension
    """
    dtype=torch.float64
    device='cpu'

    Q = torch.rand(8,8, dtype=dtype, device=device)
    R = torch.rand(4,8,8, dtype=dtype, device=device)
    beta = 10*torch.rand(1, dtype=dtype, device=device).item()
    mps = pcmps(Q, R)

    A = torch.rand(8,8, dtype=dtype, device=device)
    A = A + A.t()
    _, U = torch.symeig(A, eigenvectors=True)
    mps1 = mps.project(U)
   
    assert np.isclose(Fidelity(mps, mps1, beta), 0.5*ln_ovlp(mps, mps, beta))

