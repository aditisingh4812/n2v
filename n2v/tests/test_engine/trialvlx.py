import numpy as np
import n2v
import veloxchem as vlx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

molecule_data = """1

Ar        0.00000000    0.00000000    0.00000
"""

# Instead of passing a Molecule object, you pass the molecule_data string
molecule = vlx.Molecule.read_xyz_string(molecule_data)
print(dir(molecule))
# Now you can set the system
basis = '6-31g'
basis1 = vlx.MolecularBasis.read(molecule, basis, ostream=None)
basis2 = vlx.MolecularBasis.read(molecule, 'sto-3g', ostream=None)
print("basis1_dir",basis1)
ao_basis_map = basis1.get_ao_basis_map(molecule)
print("ao_basis_map",ao_basis_map)
label = basis1.get_avail_basis()
print("Basis set label:", label)
#nbf = basis1.get_dimension_of_basis(molecule)
#print("nbf",nbf)
ref =1
scf_drv = vlx.ScfRestrictedDriver()
scf_drv.xcfun = "slater"
grid_drv = vlx.veloxchemlib.GridDriver()
grid_level =1
grid_drv.set_level(grid_level)
molgrid = grid_drv.generate(molecule)

scf_results = scf_drv.compute(molecule, basis1)
scf_results_v = scf_drv.compute(molecule, basis2)
inv = n2v.Inverter(engine='veloxchem')
print("dir(molecule)")
# Assuming molecule.get_charge() provides a list of charges for each atom
atomic_charges = molecule.get_elemental_composition()
# Let's first check the composition to make sure it is a set
atomic_composition = molecule.get_elemental_composition()
# 'atomic_composition' is a set, for example: {8, 6}

atomic_charges = np.array([])
for element in atomic_composition:
    atomic_charges = np.append(atomic_charges,element)

print(atomic_charges)
print(atomic_charges.shape)
# Assuming that you want to get the coordinates for all atoms
atomic_coords = np.array([molecule.get_atom_coordinates(i) for i in range(molecule.number_of_atoms())])
print(f"atomic_coords shape: {atomic_coords.shape}")
print("atomic_coords",atomic_coords)


inv.set_system(molecule_data, basis,ref = ref,scf_results=scf_results)

#inv.Dt = [scf_results['D_alpha'], scf_results['D_alpha']]  # Density matrices for alpha and beta
#print("inv_Dt",inv.Dt)

#print("invers_dt_shape",np.array(inv.Dt).shape)
#inv.ct = [scf_results['C_alpha'], scf_results['C_beta']]  # Coefficients for alpha and beta
#inv.et = [scf_results['E_alpha'], scf_results['E_beta']]  # Eigenvalues for alpha and beta

inv.from_scf(molecule_data,basis,scf_result=scf_results)
# Now, you can proceed with inversion
#inv.invert("wuyang", opt_max_iter=1000, opt_method="L-BFGS-B", reg=1e-5, gtol=1e-6, guide_components="fermi_amaldi")
inv.invert("wuyang", opt_max_iter=1000, opt_method="trust-exact", reg=0, gtol=1e-6, guide_components="fermi_amaldi")
#inv.Dt = np.array(inv.Dt, dtype =float)
grid_drv = vlx.GridDriver()
# Step 2: Access grid points and weights
x_coords = molgrid.x_to_numpy()  # Get x coordinates as a NumPy array
np.save("x_coords",x_coords)
y_coords = molgrid.y_to_numpy()
np.save("y_coords",y_coords)# Get y coordinates as a NumPy array
z_coords = molgrid.z_to_numpy()  # Get z coordinates as a NumPy array
np.save("z_coords",z_coords)
coords = np.vstack((x_coords, y_coords, z_coords)).T  # Combine into a single array of coordinates
print("coords",coords)
np.save("all",coords)
# Combine the coordinates into a single array of points (spherical grid)
w = molgrid.w_to_numpy()  # Weights associated with the grid points
np.save("w",w)

D = scf_results['D_alpha'] + scf_results['D_beta']
#np.testing.assert_allclose(D_ao, D, atol=1e-12)
density = inv.eng.grid.density(scf_results['D_alpha'], scf_results['D_beta'],grid='spherical')
print("density",density)
print("inv.v_pbs",inv.v_pbs)
vrest = inv.eng.grid.to_grid(inv.v_pbs, grid='spherical')
print("vrest",vrest)

np.save("vrest",vrest)
vext = inv.eng.grid.external( grid='spherical')
print("vext",vext)
vH = inv.eng.grid.hartree(density=scf_results['D_alpha'] + scf_results['D_beta'])
#vH_1 = vlx.ao_matrix_to_dalton(vlx.DenseMatrix(vH),
#                                    basis1, molecule).to_numpy()
#print("vH1",vH1)
print("vH",vH)

vFA = (1 - 1/(inv.nalpha + inv.nbeta)) * vH
print("vFA",vFA)

vxc = vFA + vrest - vH
print("vxc",vxc)
np.save("vxc",vxc)

grid_points = inv.eng.grid.spherical_points  # Shape (N, 3), where N is the number of grid points
x, y, z = grid_points[:, 0], grid_points[:, 1], grid_points[:, 2]

potential = vH.flatten()  # Ensure it's 1D

import matplotlib.pyplot as plt
import numpy as np

# Filter points where X is non-negative
mask = x >= 0
x_filtered = x[mask]
potential_filtered = potential[mask]

plt.figure(figsize=(8, 6))
plt.scatter(x_filtered, potential_filtered, c=potential_filtered, cmap='coolwarm', marker='o')
plt.colorbar(label="Potential (Hartree)")
plt.title("Potential Along X-Axis (X ≥ 0)")
plt.xlabel("X")
plt.ylabel("Potential (Hartree)")
plt.grid(True)

# Save the figure
plt.savefig("potential_vs_x_positive.png", dpi=300, bbox_inches='tight')
plt.show()



exit()
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')

# Choose potential to plot (change vrest to vext, vH, etc.)
potential = vH.flatten()  # Ensure it's 1D

sc = ax.scatter(x, y, z, c=potential, cmap='coolwarm', marker='o')
plt.colorbar(sc, label="Potential (Hartree)")
ax.set_title("Rest Potential on Spherical Grid")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()
plt.savefig("rest_potential_3D.png", dpi=300, bbox_inches='tight')

exit()
vext = inv.eng.grid.external( grid='spherical') 

vH = inv.eng.grid.hartree(density=scf_results['D_alpha'])
#vH_1 = vlx.ao_matrix_to_dalton(vlx.DenseMatrix(vH),
#                                    basis1, molecule).to_numpy()
#print("vH1",vH1)
print("vH",vH)
print("shape of vh", vH.shape)
print("inv.nalpha",inv.nalpha)
print("inv.nbeta",inv.nbeta)
vFA = (1 - 1/(inv.nalpha + inv.nbeta)) * vH
print("vFA",vFA)
print("inv.v_pbs",inv.v_pbs)
vrest = inv.eng.grid.to_grid(inv.v_pbs, grid='spherical')
print("vrest",vrest)

vxc = vFA + vrest - vH
print("vxc",vxc)
np.save("vxc",vxc)
# Load the coordinates
x_coords = np.load("x_coords.npy")
y_coords = np.load("y_coords.npy")
z_coords = np.load("z_coords.npy")
vxc = np.load("vxc.npy")  # Assuming you've stored vxc values

# Create 3D scatter plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with color mapping
sc = ax.scatter(x_coords, y_coords, z_coords, c=vxc, cmap="viridis", marker="o")

# Colorbar for potential values
plt.colorbar(sc, ax=ax, label=r'$v_{xc}$')

# Labels and title
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Exchange-Correlation Potential $v_{xc}$ on Spherical Grid")
# Save as PNG
plt.savefig("vxc_plot.png", dpi=300, bbox_inches='tight')

plt.show()

