<div align="center">
<img align="middle" src="_assets/logo.png" width="500" alt="logo"/>
<h2> Continuous Matrix Product Operator (cMPO) for Quantum Thermodynamics </h2>
</div>

[![Build Status](https://api.travis-ci.com/TensorBFS/cMPO.svg?token=bCr4MXNqz8r1WxLSWzAC&branch=master)](https://travis-ci.com/github/TensorBFS/cMPO)

This is a PyTorch implementation of the cMPO approach to finite temperature quantum states. The cMPO approach is applicable to one-dimensional quantum systems with short-range or long-range interactions. This approach is described in the paper "Continuous Matrix Product Operator Approach to Finite Temperature Quantum States" (arXiv link).

## Features

- **Say NO to Trotter error:** thanks to the coutinuous-time formulation 
- **Simple yet generic:**  a unified interface to any Hamiltonian with an MPO representation

## Bonus 

- **Real-frequency local spectral functions:** analytic continuation is joyful, finally

## Example

```bash
python power_projection.py -bondD 10 -beta 10 -Gamma 1 -J 1 -resultdir isingdata
```

we calculate the thermodynamics of the transverse field Ising model 
$$
H=-J\sum_{\langle i,j\rangle} Z_i Z_j -\Gamma\sum_i X_i,
$$
with $J=1,\Gamma=1$ at temperature $\beta=10$ using cMPO-cMPS method with bond dimension $\chi=10$. The calculation results, including the free energy, internal energy, specific heat, and the local susceptibility, along with the checkpoint data files, are automatically saved in the directory `isingdata`.

More models are defined in `model.py`, and one can investigate the thermodynamical properties of these models by modifying `power_projection.py` accordingly. To do this, in  `power_projection.py`, find the part `construct cMPO`

```python
    s = model.spin_half(dtype, device)
    ising = model.ising(Gamma=args.Gamma, J=args.J, dtype=dtype, device=device) # cmpo def
    T = ising.T
    W = ising.W
    ph_leg = ising.ph_leg
```

and simply replace `ising` by the model that you are interested in. You can also  easily simulate your own models by adding them to  `model.py`.

To compute more quantities, like local observables and dynamical properties, you can use `postprocess.py` to access the checkpoint data files and calculate these quantities without running your simulation again.

