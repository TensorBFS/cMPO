""" calculate additional quantites using saved data 
"""

import numpy as np
import torch
import os, sys, io, subprocess
import os.path
os.environ['OMP_NUM_THREADS']='8'
torch.set_num_threads(8)
torch.manual_seed(42)

from cmpo import *
import model

from power_projection import *

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

    bondD = int(args.data.split('bondD')[1].split('_beta')[0])
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
    Q = torch.rand(bondD, dtype=dtype, device=device)
    R = torch.rand(d,bondD,bondD, dtype=dtype, device=device)
    Q = torch.nn.Parameter(Q)
    R = torch.nn.Parameter(R)
    cmpsdata = data_cmps(Q, R)
    dataload(cmpsdata, cmpsdata_name)
    psi = cmps(torch.diag(Q), R).detach()

    Lpsi = multiply(W, psi)
    Z_value = AL_entropy(psi, Lpsi, T, s.Z, beta)

    out = args.out
    f_out = io.open(out, 'a')
    f_out.write('{:.4f}  {:.12f} \n'.format(1/beta, Z_value))
    f_out.close()

