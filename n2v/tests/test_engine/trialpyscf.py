from pyscf import gto
import matplotlib.pyplot as plt
import numpy as np
import n2v

import matplotlib as mpl
mpl.rcParams["font.size"] = 11
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["axes.edgecolor"] = "#eae8e9" 
# Define Molecule
mol = gto.M(atom = """
                  Ne
                  """,
basis = 'cc-pvdz')

# Perform Calculation
mf = mol.KS()
mf.xc = 'scan'
mf.kernel()

# Extract data for n2v.
da, db = mf.make_rdm1()/2, mf.make_rdm1()/2
ca, cb = mf.mo_coeff[:,:mol.nelec[0]], mf.mo_coeff[:, :mol.nelec[1]]
ea, eb = mf.mo_energy, mf.mo_energy

# Initialize inverter object.
inv = n2v.Inverter( engine='pyscf' )

inv.set_system( mol, 'cc-pvdz', pbs='cc-pvqz' ,ref =1)
inv.Dt = [da, db]
inv.ct = [ca, cb]
inv.et = [ea, eb]
# Inverter with WuYang method, guide potention v0=Fermi-Amaldi
inv.invert("WuYang", opt_max_iter=100, opt_method="trust-exact", gtol=1e-4,
           guide_components="fermi_amaldi")

# Build Grid
inv.eng.grid.build_rectangular((5001,1,1))
x = inv.eng.grid.x

#Compute components
vH = inv.eng.grid.hartree(density=da+db, grid='rectangular')
vFA = (1 - 1/(inv.nalpha + inv.nbeta)) * vH
vrest = inv.eng.grid.to_grid(inv.v_pbs, grid='rectangular')

# Generate inverted vxc
vxc = vFA + vrest - vH

fig, ax = plt.subplots()
ax.plot(x, vxc, label="$v(\mathbf{r})$")
ax.legend()
ax.set_xscale('log')
fig.show()
