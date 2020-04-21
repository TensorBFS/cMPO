# Continuous Matrix Product Operator (cMPO)

This repository includes the python3 + pytorch implementation of the cMPO approach to finite temperature quantum states. This approach is applicable to one-dimensional quantum systems with short-range or long-range interactions. This approach is described in the paper "Continuous Matrix Product Operator Approach to Finite Temperature Quantum States" (arXiv link).

## Usage

As an example, by 

```bash
python cmpo_1Dlongrange_TMRG.py -chi 10 -beta 10 -Gamma 1 -J 1 -resultdir isingdata
```

we calculate the thermodynamics of the transverse field Ising model 
$$
H=-J\sum_{\langle i,j\rangle} Z_i Z_j -\Gamma\sum_i X_i,
$$
with $J=1,\Gamma=1$ at temperature $\beta=10$ using cMPO-cMPS method with bond dimension $\chi=10$. The calculation results, including the free energy, internal energy, specific heat, and the local susceptibility, along with the checkpoint data files, are automatically saved in the directory `isingdata`.

More models are defined in `model_1DLR.py`, and one can investigate the thermodynamical properties of these models by modifying `cmpo_1Dlongrange_TMRG.py` accordingly. To do this, in  `cmpo_1Dlongrange_TMRG.py`, find the part `construct cMPO`

```python
    s = model.spin_half(dtype, device)
    ising = model.ising(Gamma=Gamma, J=J, dtype=dtype, device=device) # cmpo def
    T = ising.T
    W = ising.W
    ph_leg = ising.ph_leg
```

and simply replace `ising` by the model that you are interested in. You can also  easily simulate your own models by adding them to  `model_1DLR.py`.

To compute more quantities, like local observables and dynamical properties, one can use `cmpo_1Dlongrange_read.py` to access the checkpoint data files and calculate these quantities without running your simulation again.

