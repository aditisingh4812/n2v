import numpy as np
import n2v
import veloxchem as vlx
import matplotlib.pyplot as plt

molecule_data = """2

H    0.000000000000        0.740848095288        0.582094932012
H    0.000000000000       -0.740848095288        0.582094932012
"""

# Instead of passing a Molecule object, you pass the molecule_data string
molecule = vlx.Molecule.read_xyz_string(molecule_data)

# Now you can set the system
basis = 'sto-3g'
basis = vlx.MolecularBasis.read(molecule, basis, ostream=None)
ref =1
scf_drv = vlx.ScfRestrictedDriver()
scf_drv.ostream.mute()
scf_results = scf_drv.compute(molecule, basis)
basis = 'sto-3g'
inv = n2v.Inverter(engine='veloxchem')
# Now, pass scf_results to set_system
inv.set_system(molecule_data, basis, ref=ref, pbs='same', scf_results=scf_results)

# Now you can proceed with the inversion or other methods
inv.invert("WuYang", opt_max_iter=100, opt_method="trust-exact", reg=0, gtol=1e-6, guide_components="fermi_amaldi")

exit()
# Mock some example data for testing (replace these with real calculation data)
# In a real calculation, this would be extracted from the results of the SCF procedure.
da = np.random.rand(5, 5)  # Mock density matrix
ca = np.random.rand(5, 5)  # Mock coefficient matrix
ea = np.random.rand(5)     # Mock eigenvalues (orbital energies)

# Set the inverter data
inv.Dt = [da, da]  # Mock density matrices (should be split for alpha/beta)
inv.ct = [ca, ca]  # Mock coefficients
inv.et = [ea, ea]  # Mock eigenvalues

# Perform the inversion
inv.invert("WuYang", opt_max_iter=100, opt_method="trust-exact", reg=0, gtol=1e-6, guide_components="fermi_amaldi")

# Build the grid for potential calculations
x = np.linspace(-5, 5, 43)
y = np.zeros_like(x)
z = np.zeros_like(x)
grid = np.array([x, y, z])

# Generate the grid for the calculation
grid2 = inv.eng.grid.generate_grid(x=x, y=[0], z=[0])[0]

# Compute the Hartree potential and Fermi-Amaldi potential
vH1 = inv.eng.grid.esp(Da=inv.Dt[0], Db=inv.Dt[1], grid=grid)[1]
vFA1 = (1 - 1 / (inv.nalpha + inv.nbeta)) * vH1

# Compute the rest potential
vrest1 = inv.eng.grid.ao(inv.v_pbs, grid=grid, basis=inv.eng.pbs)

# Compute the exchange-correlation potential
vxc1 = vFA1 + vrest1 - vH1

# Save results to disk (optional)
np.save("dm.npy", da)
np.save("coeff.npy", ca)
np.save("eigen.npy", ea)

# Write the results to a file
with open('file.txt', 'w') as f:
    s = 0
    for x_value in x:
        y = round(x_value, 6)
        f.write(f"{y}\t\t\t{vxc1[s]}\n")
        s += 1

# Inversion for new data (adjust the density matrix, coefficients, etc.)
da1 = da + 1e-7 * np.eye(da.shape[0])  # Adjust density matrix
new_da_matrix = vlx.core.Matrix.from_array(da1)
np.save("dm1.npy", new_da_matrix)

# Overwrite the existing wavefunction with the new values
wfn.Da().copy(new_da_matrix)
wfn.Ca().copy(vlx.core.Matrix.from_array(ca))

# Perform inversion with updated data
inv.invert("WuYang", opt_max_iter=100, opt_method="SLSQP", reg=0, gtol=1e-6)

# Compute the new vxc (exchange-correlation) potential using updated data
vH2 = inv.eng.grid.esp(Da=inv.Dt[0], Db=inv.Dt[1], grid=grid)[1]
vFA2 = (1 - 1 / (inv.nalpha + inv.nbeta)) * vH2
vrest2 = inv.eng.grid.ao(inv.v_pbs, grid=grid, basis=inv.eng.pbs)

# Compute the new vxc
vxc2 = vFA2 + vrest2 - vH2
vxc_diff = vxc1 - vxc2

# Write the vxc differences to a file
with open('file.txt', 'a') as f:
    s = 0
    for x_value in x:
        f.write(f"{round(x_value, 6)}\t\t\t{vxc_diff[s]}\n")
        s += 1

