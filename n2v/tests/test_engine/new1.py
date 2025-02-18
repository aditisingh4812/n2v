import numpy as np
import veloxchem as vlx

# Corrected multi-line string format
h2o_xyz = """1

Ne        0.00000000    0.00000000    0.00000
"""

# Define molecule and basis set
molecule = vlx.Molecule.read_xyz_string(h2o_xyz)  # Corrected method call
basis = vlx.MolecularBasis.read(molecule, "sto-3g")

# Run SCF calculation
scf_drv = vlx.ScfRestrictedDriver()
scf_drv.ostream.mute()
scf_results = scf_drv.compute(molecule, basis)

# Compute total density matrix
D = scf_results["D_alpha"] + scf_results["D_beta"]

# Load positions from file
positions = np.load('all.npy')  # Ensure the file exists
print("positions:", positions)
weights = np.load('w.npy')
# Define charges
charges = -1.0 * np.ones(len(positions))
print("charges:", charges)

# Compute nuclear potential integrals
pot_drv = vlx.NuclearPotentialIntegralsDriver()
v_c = {}

for charge, position in zip(charges, positions):
    position_tuple = tuple(position)  # Convert to tuple for dictionary key
    v_np = -1.0 * pot_drv.compute(molecule, basis, [charge], [position]).to_numpy()
    v_c[position_tuple] = np.einsum("ab, ab ->", D, v_np)
    
# Print results
for position, value in v_c.items():
    print( value)

