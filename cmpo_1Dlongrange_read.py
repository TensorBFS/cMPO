""" calculate additional quantites using saved data 
"""

import numpy as np
import torch
import os, sys, io, subprocess
import os.path
os.environ['OMP_NUM_THREADS']='8'
torch.set_num_threads(8)
torch.manual_seed(42)

from pcmpo_1Dlongrange import pcmpo, pcmps, multiply, act, ln_ovlp, density_matrix, mera_update, self_compress, data_cmps, datasave, dataload
import model_1DLR as model

import cmpo_1Dlongrange_TMRG as tmrg

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-data", type=str, default='none', help="data")
    parser.add_argument("-out", type=str, default='out.dat', help="data")

    parser.add_argument("-float32", action='store_true', help="use float32")
    parser.add_argument("-cuda", type=int, default=-1, help="use GPU")
    args = parser.parse_args()
    device = torch.device("cpu" if args.cuda<0 else "cuda:"+str(args.cuda))
    dtype = torch.float32 if args.float32 else torch.float64

    chi = int(args.data.split('_chi')[1].split('_beta')[0])
    beta = float(args.data.split('_beta')[1].split('_Gamma')[0])
    Gamma = float(args.data.split('_Gamma')[1].split('_J')[0])
    J = float(args.data.split('_J')[1].split('-meas')[0])

    s = model.spin_half(dtype, device)
    ising = model.ising(Gamma=Gamma, J=J, dtype=dtype, device=device)
    T = ising.T
    W = ising.W
    d = ising.d
    ph_leg = ising.ph_leg

    measdata_name = args.data
    f_meas = io.open(measdata_name, 'r')
    data_arr = np.loadtxt(f_meas)
    optim_step = int(data_arr[np.argmin(data_arr[:, 1]), 0])
    print('read data from step %g'%optim_step)

    cmpsdata_name = args.data[:-9]+'/psi_{:03d}.pt'.format(optim_step)
    Q = torch.rand(chi, dtype=dtype, device=device)
    R = torch.rand(d,chi,chi, dtype=dtype, device=device)
    Q = torch.nn.Parameter(Q)
    R = torch.nn.Parameter(R)
    cmpsdata = data_cmps(Q, R)
    dataload(cmpsdata, cmpsdata_name)
    psi = pcmps(torch.diag(Q), R).detach()

    #chi_loc = tmrg.chi(psi, W, T, s.Z, s.Z, beta) / beta
    wrange = np.linspace(-0.1, 0.1, 21) + 1e-6
    S0 = 0.01*np.sum([tmrg.spectral(psi, W, T, s.Z, s.Z, beta, omega, eta=0.05) for omega in wrange])

    out = args.out
    f_out = io.open(out, 'a')
    f_out.write('{:.4f}  {:.12f} \n'.format(1/beta, S0))
    f_out.close()

#    import matplotlib 
#    import matplotlib.pyplot as plt
#    plt.yscale('log')
#    plt.xlabel(r'$\omega$')
#    plt.ylabel(r'$S_0$')

#    omega_list = np.linspace(-0.1,0.1,101)+1e-6 
#    for eta in [0.1, 0.05, 0.01, 0.001]:
#        S_list = [tmrg.spectral(psi, W, T, s.Z, s.Z, beta, omega, eta) for omega in omega_list]
#        plt.plot(omega_list, np.abs(S_list), '-', markerfacecolor='none', label=r'$\eta={:.3f}$'.format(eta))
#    plt.legend()
#    plt.savefig('hello.pdf')

