""" main program: power projection procedure
"""

import numpy as np
import torch
import os, sys, io, subprocess
import os.path
torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))
torch.manual_seed(42)

from cmpo import *
import model

def F(psi, Lpsi, T, beta):
    """ calculate the free energy by
            -(1/beta) * log [<Lpsi|T|psi> / <Lpsi|psi>]
        T: cMPO
        psi: cMPS (right eigenvector)
        Lpsi: cMPS (left eigenvector)
        beta: inverse temperature
    """
    Tpsi = act(T, psi)
    up = ln_ovlp(Lpsi, Tpsi, beta)
    dn = ln_ovlp(Lpsi, psi, beta)
    return (- up + dn) / beta

def Obsv(psi, Lpsi, T, O, beta):
    """ calculate the thermal average of the observable O as
            <I \otimes O \otimes I>_{K}
        where K is the K-matrix corresponding to <Lpsi|T|psi>
        K plays the role of effective Hamiltonian
        T: cMPO
        psi: cMPS (right eigenvector)
        Lpsi: cMPS (left eigenvector)
        O: the observable
        beta: inverse temperature
    """
    dtype, device = psi.dtype, psi.device
    totalD = O.shape[0]*psi.dim*psi.dim
    matI = torch.eye(psi.dim, dtype=dtype, device=device)
    matO = torch.einsum('ab,cd,ef->acebdf', matI, O, matI).contiguous().view(totalD, totalD) 
 
    Tpsi = act(T, psi)
    M = density_matrix(Lpsi, Tpsi)

    w, v = eigensolver(M)
    w -= w.max().item()
    expw = torch.diag(torch.exp(beta*w))
    return torch.trace(expw @ v.t() @ matO @ v).item() / torch.trace(expw).item()

def Corr(psi, Lpsi, T, O1, O2, beta, tau):
    """ calculate unequal-imaginary-time correlator
            tr[exp(-beta*K) O1(0) O2(tau)]
        where K is the K-matrix corresponding to <Lpsi|T|psi>
        K plays the role of effective Hamiltonian
        T: cMPO
        psi: cMPS (right eigenvector)
        Lpsi: cMPS (left eigenvector)
        O1, O2: observables
        beta: inverse temperature
    """
    dtype, device = psi.dtype, psi.device
    totalD = O1.shape[0]*psi.dim*psi.dim
    matI = torch.eye(psi.dim, dtype=dtype, device=device)
    matO1 = torch.einsum('ab,cd,ef->acebdf', matI, O1, matI).contiguous().view(totalD, totalD) 
    matO2 = torch.einsum('ab,cd,ef->acebdf', matI, O2, matI).contiguous().view(totalD, totalD) 
    Tpsi = act(T, psi)
    M = density_matrix(Lpsi, Tpsi)

    w, v = eigensolver(M)
    matO1 = v.t() @ matO1 @ v
    matO2 = v.t() @ matO2 @ v
    w -= w.max().item()

    expw_a = torch.diag(torch.exp((beta-tau)*w))
    expw_b = torch.diag(torch.exp(tau*w))
    expw = torch.diag(torch.exp(beta*w))
    return torch.trace(expw_a @ matO1 @ expw_b @ matO2).item() / torch.trace(expw).item()

def chi(psi, Lpsi, T, O1, O2, beta, iomega=0):
    """ calculate the local susceptibility with
            \int_0^{\beta} d tau tr[exp(-beta*K) O1(0) O2(tau)]
        where K is the K-matrix corresponding to <Lpsi|T|psi>
        K plays the role of effective Hamiltonian
        T: cMPO
        psi: cMPS (right eigenvector)
        Lpsi: cMPS (left eigenvector)
        O1, O2: observables
        beta: inverse temperature
    """
    dtype, device = psi.dtype, psi.device
    totalD = O1.shape[0]*psi.dim*psi.dim
    matI = torch.eye(psi.dim, dtype=dtype, device=device)
    matO1 = torch.einsum('ab,cd,ef->acebdf', matI, O1, matI).contiguous().view(totalD, totalD) 
    matO2 = torch.einsum('ab,cd,ef->acebdf', matI, O2, matI).contiguous().view(totalD, totalD) 
    Tpsi = act(T, psi)
    M = density_matrix(Lpsi, Tpsi)

    w, v = eigensolver(M)
    w -= w.max().item()
    matO1 = v.t() @ matO1 @ v
    matO2 = v.t() @ matO2 @ v
    expw = torch.diag(torch.exp(beta*w))
    
    Envec, Emvec = np.meshgrid(w.detach().numpy(), (w+1e-8).detach().numpy())
    F = (np.exp(beta*Envec) - np.exp(beta*Emvec)) / (iomega + Envec - Emvec)
    F = torch.tensor(F, dtype=dtype, device=device)

    result = torch.trace(matO1 @ (matO2 * F)) / torch.trace(expw)
    return result.item()

def chi2(psi, Lpsi, T, O1, O2, beta, omega=0, eta=0.05):
    dtype, device = psi.dtype, psi.device
    totalD = O1.shape[0]*psi.dim*psi.dim
    matI = torch.eye(psi.dim, dtype=dtype, device=device)
    matO1 = torch.einsum('ab,cd,ef->acebdf', matI, O1, matI).contiguous().view(totalD, totalD) 
    matO2 = torch.einsum('ab,cd,ef->acebdf', matI, O2, matI).contiguous().view(totalD, totalD) 
    Tpsi = act(T, psi)
    M = density_matrix(Lpsi, Tpsi)

    w, v = eigensolver(M)
    w -= w.max().item()
    matO1 = v.t() @ matO1 @ v
    matO2 = v.t() @ matO2 @ v
    expw = torch.diag(torch.exp(beta*w))

    delta = lambda x, eta: 1/np.pi * eta/(x**2 + eta**2)
    Envec, Emvec = np.meshgrid(w.detach().numpy(), w.detach().numpy())
    F = (np.exp(beta*Envec) - np.exp(beta*Emvec)) * delta(omega + Envec - Emvec, eta)
    F = torch.tensor(F, dtype=dtype, device=device)

    result = -np.pi * torch.trace(matO1 @ (matO2 * F)) / torch.trace(expw)
    return result.item()

def spectral(psi, W, T, O1, O2, beta, omega=1e-6, eta=0.05):
    """ calculate spectral function
    """
    return 2*chi2(psi, W, T, O1, O2, beta, omega, eta) / (1-np.exp(-beta*omega))

def E(psi, Lpsi, T, beta):
    """ calculate the energy 
        T: cMPO
        psi: cMPS (right eigenvector)
        Lpsi: cMPS (left eigenvector)
        beta: inverse temperature
    """
    Tpsi = act(T, psi)
    M = density_matrix(Lpsi, Tpsi)
    w, _ = eigensolver(M)
    w_nm = w - torch.logsumexp(beta * w, dim=0)/beta
    up = torch.exp(beta * w_nm) @ w 

    M = density_matrix(Lpsi, psi)
    w, _ = eigensolver(M)
    w_nm = w - torch.logsumexp(beta * w, dim=0)/beta
    dn = torch.exp(beta * w_nm) @ w 

    return (- up + dn).item()

def Cv(psi, Lpsi, T, beta):
    """ calculate the specific heat 
        T: cMPO
        psi: cMPS (right eigenvector)
        Lpsi: cMPS (left eigenvector)
        beta: inverse temperature
    """
    Tpsi = act(T, psi)
    M = density_matrix(Lpsi, Tpsi)
    w, _ = eigensolver(M)
    w_nm = w - torch.logsumexp(beta * w, dim=0)/beta
    up = torch.exp(beta * w_nm) @ (w*w) - (torch.exp(beta * w_nm) @ w)**2

    M = density_matrix(Lpsi, psi)
    w, _ = eigensolver(M)
    w_nm = w - torch.logsumexp(beta * w, dim=0)/beta
    dn = torch.exp(beta * w_nm) @ (w*w) - (torch.exp(beta * w_nm) @ w)**2

    return (beta**2) * (up - dn).item()

def reduced_density_matrix(psi, tau, beta):
    bondD = psi.dim
    M = density_matrix(psi, psi)
    w0, v0 = eigensolver(M)
    w0 = w0 - torch.logsumexp(beta*w0, dim=0) / beta
    expM = lambda x: v0 @ torch.diag(torch.exp(x*w0)) @ v0.t()
    def reshapeM(M): 
        M1 = M.reshape(bondD, bondD, bondD, bondD)
        M1 = M1.permute(0,2,1,3).reshape(bondD**2, bondD**2)
        return M1
    expM1 = reshapeM(expM(tau))
    expM2 = reshapeM(expM(beta - tau))

    w, v = eigensolver(expM1)
    w_sqrt = torch.diag(torch.sqrt(w))    

    return w_sqrt @ v.t() @ expM2.t() @ v @ w_sqrt

def entanglement_entropy(psi, tau, beta):
    """ use the boundary cMPS to calculate von Neumann entropy
        on interval tau
    """
    rho = reduced_density_matrix(psi, tau, beta)
    w, _ = eigensolver(rho)

    result = -torch.sum(w * torch.log(w))
    # fot test (renyi-2)
    #result = torch.log(torch.sum(w**2)) / (1-2)
    return result

def klein(psi, W, beta):
    """ calculate the klein bottle entropy
        psi: cMPS (right eigenvector)
        W: transform psi to the left eigenvector
        beta: inverse temperature
    """
    bondD = psi.dim

    Wpsi = multiply(W, psi)
    M = density_matrix(psi, Wpsi)
    w, v = eigensolver(M)
    w_nm = w - torch.logsumexp(beta * w, dim=0)/beta
    
    exp_M = v @ torch.diag(torch.exp(0.5*beta*w_nm)) @ v.t()
    gk = torch.einsum('abba->', exp_M.reshape(bondD,bondD,bondD,bondD))
    sk = 2*torch.log(gk)

    return sk.item() 

def effectiveH(psi, Lpsi, num):
    """ calculate the spectrum of the effective Hamiltonian
        psi: cMPS (right eigenvector)
        Lpsi: cMPS (left eigenvector)
        num: number of states
    """
    H = density_matrix(Lpsi, psi)

    w, _ = eigensolver(H)
    return -w[-num:] + w[-1]

def name_gen(args):
    """ generate the name for the datafile with the parameters 
    """
    fl_to_str = lambda x: '{:.4f}'.format(x)
    s = ''
    s += 'bondD' + str(args.bondD) 
    s += '_beta' + fl_to_str(args.beta)
    s += '_Gamma' + fl_to_str(args.Gamma)
    s += '_J' + fl_to_str(args.J)
    return s

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-bondD", type=int, default=2, help="tau direction bond dimension")
    parser.add_argument("-beta", type=float, default=4.0 , help="beta")
    parser.add_argument("-Gamma", type=float, default=1.0, help="Gamma")
    parser.add_argument("-J", type=float, default=1.0, help="J")
    parser.add_argument("-alpha", type=float, default=1.0, help="alpha")
    parser.add_argument("-Nmax", type=int, default=6, help="Nmax")

    parser.add_argument("-resultdir", type=str, default='tmpdata', help="result folder")
    parser.add_argument("-init", type=str, default='none', help="init")

    parser.add_argument("-float32", action='store_true', help="use float32")
    parser.add_argument("-cuda", type=int, default=-1, help="use GPU")
    args = parser.parse_args()
    device = torch.device("cpu" if args.cuda<0 else "cuda:"+str(args.cuda))
    dtype = torch.float32 if args.float32 else torch.float64
    bondD = args.bondD
    beta = args.beta

    ### datafile
    if (not os.path.exists(args.resultdir)):
        print("---> creating directory to save the result: " + args.resultdir)
    key = args.resultdir+'/'+name_gen(args)

    cmd = ['mkdir', '-p', key]
    subprocess.check_call(cmd)
    print("---> log and chkp saved to ", key)

    ### construct cMPO for more models see model.py
    s = model.spin_half(dtype, device)
    ising = model.ising(Gamma=args.Gamma, J=args.J, dtype=dtype, device=device) # cmpo def
    T = ising.T
    W = ising.W
    ph_leg = ising.ph_leg
    d = ising.d

    ### initialize
    if args.init == 'none':
        ### after initialization bondD/ph_leg < psi.dim < or = bondD
        init_step = int(np.floor(np.log(bondD) / np.log(ph_leg)))
        psi = cmps(T.Q, T.L)
        Lpsi = cmps(T.Q, T.R)
        for ix in range(init_step-1):
            psi = act(T, psi)  
            Lpsi = act(T.t(), psi)
        psi = psi.diagQ()
        Lpsi = Lpsi.diagQ()
    else:
        f_meas = io.open(args.init, 'r')
        data_arr = np.loadtxt(f_meas)
        optim_step = int(data_arr[np.argmin(data_arr[:, 1]), 0])
        print('optim step', optim_step)
    
        D0 = int(args.init.split('bondD')[1].split('_beta')[0])
        beta0 = float(args.init.split('_beta')[1].split('_Gamma')[0])

        # read out right eigenvector
        psidata_name = args.init[:-9]+'/psi_{:03d}.pt'.format(optim_step)
        Q = torch.rand(D0, dtype=dtype, device=device)
        R = torch.rand(d,D0,D0, dtype=dtype, device=device)
        Q = torch.nn.Parameter(Q)
        R = torch.nn.Parameter(R)
        psidata = data_cmps(Q, R)
        dataload(psidata, psidata_name)
        factor = beta0/beta
        psi = cmps(torch.diag(Q)*factor, R*np.sqrt(factor)).detach()

        # read out left eigenvector
        if W is None:
            Lpsidata_name = args.init[:-9]+'/Lpsi_{:03d}.pt'.format(optim_step)
            Ql = torch.rand(D0, dtype=dtype, device=device)
            Rl = torch.rand(d,D0,D0, dtype=dtype, device=device)
            Ql = torch.nn.Parameter(Ql)
            Rl = torch.nn.Parameter(Rl)
            Lpsidata = data_cmps(Ql, Rl)
            dataload(Lpsidata, Lpsidata_name)
            Lpsi = cmps(torch.diag(Ql), Rl).detach()
        else:
            Lpsi = multiply(W, psi)

    # power procedure 
    power_counter, step = 0, 0
    Fmin = 9.9e9
    while power_counter < 3:
        if psi.dim <= bondD:
            Tpsi = act(T, psi)
        else:
            Tpsi = psi
        psi = variational_compr(Tpsi, beta, bondD, chkp_loc=key+'/psi_{:03d}.pt'.format(step))

        if W is None: # this part is outdated
            LpsiT = act(T.t(), Lpsi)
            Lpsi = variational_compr(LpsiT, beta, bondD, chkp_loc=key+'/Lpsi_{:03d}.pt'.format(step))
        else:
            Lpsi = multiply(W, psi)

        # output measurements
        F_value = F(psi, Lpsi, T, beta)
        E_value = E(psi, Lpsi, T, beta)
        Cv_value = Cv(psi, Lpsi, T, beta)
        Sk_value = klein(psi, W, beta)
        #chi_value = chi(psi, Lpsi, T, s.Z/2, s.Z/2, beta) / beta

        # output measurement results
        logfile_meas = io.open(key+'-meas.log', 'a')
        message = ('{} ' + 4*'{:.12f} ').format(step, F_value, E_value, Cv_value, Sk_value)
        print('step, F, E, Cv, Sk '+ message, end='  \n')
        logfile_meas.write(message + u'\n')
        logfile_meas.close()

        step += 1
        if F_value < Fmin - 1e-11:
            power_counter = 0
            Fmin = F_value
        else: 
            power_counter += 1
    print('  ')

