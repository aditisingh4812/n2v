import numpy as np
import veloxchem as vlx

# Define the molecule
ne_xyz = """1

Ne   0.00000000     0.00000000    0.0
"""

molecule = vlx.Molecule.read_xyz_string(ne_xyz)
basis = vlx.MolecularBasis.read(molecule, "6-31g")

# Run SCF calculation
scf_drv = vlx.ScfRestrictedDriver()
scf_drv.ostream.mute()
scf_results = scf_drv.compute(molecule, basis)

# Extract density matrix
D = scf_results["D_alpha"] + scf_results["D_beta"]

# Load positions from file
positions = np.load('all.npy')  # Ensure this file exists and contains valid positions
print("Positions:", positions)

import numpy as np
import veloxchem as vlx


scf_drv = vlx.ScfRestrictedDriver()
scf_drv.xcfun = "slater"
grid_drv = vlx.veloxchemlib.GridDriver()
grid_level =1
grid_drv.set_level(grid_level)
molgrid = grid_drv.generate(molecule)
x_coords = molgrid.x_to_numpy()  # Get x coordinates as a NumPy array
y_coords = molgrid.y_to_numpy()  # Get y coordinates as a NumPy array
z_coords = molgrid.z_to_numpy()  # Get z coordinates as a NumPy array
positions = np.vstack((x_coords, y_coords, z_coords)).T  # Combine into a single array of coordinates


# Define positions as a 3D grid
z = np.load("z_coords.npy")#np.linspace(-5, 5, 20)
y = np.load("y_coords.npy")#np.zeros_like(z)
x = np.load("x_coords.npy")#np.zeros_like(z)
positions = np.vstack((x, y, z)).T  # Shape (43, 3), ensuring 3D positions

# Define charges
charges = -1.0 * np.ones(len(positions))  # Charge for each position
print("Charges:", charges)

# Compute nuclear potential integrals
pot_drv = vlx.NuclearPotentialIntegralsDriver()
hartree_potential = np.zeros(len(positions))
v_c = {}

for i, (charge, position) in enumerate(zip(charges, positions)):
    v_np = -pot_drv.compute(molecule, basis, [charge], [position]).to_numpy()
    hartree_potential[i] = np.einsum("ab, ab ->", D, v_np)
    v_c[tuple(position)] = hartree_potential[i]  # Dictionary assignment corrected

print("Hartree Potential:", hartree_potential)

import numpy as np
import matplotlib.pyplot as plt



vH1 = np.load("vh.npy")#inv.eng.grid.esp(Da=inv.Dt[0], Db=inv.Dt[1], grid=grid)[1]
print("vH1:", vH1)

vrest = np.load("vrest.npy")
vrest1 = np.load("vrest1.npy")

# Plot both potentials
plt.figure(figsize=(8, 5))
plt.plot(positions, vrest, marker='o', linestyle='-', color='b', label=r'veloxchem')
plt.plot(positions, vrest1, marker='s', linestyle='--', color='r', label=r'psi4')

# Labels and title
plt.xlabel('Z (Bohr)')
plt.ylabel('Hartree Potential')
plt.title('Hartree Potentials for CO')
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)  # Reference line at y=0
plt.legend()
plt.grid(True)

# Save the figure
plt.savefig("potentials_comparison.png", dpi=300, bbox_inches='tight')
# Show the figure
plt.show()

