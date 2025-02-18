import numpy as np
import veloxchem as vlx

# Define molecule
h2o_xyz = """1

Ne        0.00000000    0.00000000    0.00000
"""

molecule = vlx.Molecule.read_xyz_string(h2o_xyz)
basis = vlx.MolecularBasis.read(molecule, "sto-3g")

# Run SCF calculation
scf_drv = vlx.ScfRestrictedDriver()
scf_drv.ostream.mute()
scf_results = scf_drv.compute(molecule, basis)

# Compute total density matrix
D = scf_results["D_alpha"] + scf_results["D_beta"]

# Load positions and weights
positions = np.load('all.npy')  # Ensure the file exists
weights = np.load('w.npy')      # Ensure the file exists
print("positions:", positions)
print("weights:", weights)

# Define charges
charges = -1.0 * np.ones(len(positions))
print("charges:", charges)

# Compute nuclear potential integrals
pot_drv = vlx.NuclearPotentialIntegralsDriver()
v_c = {}

for charge, position, weight in zip(charges, positions, weights):
    position_tuple = tuple(position)  # Convert to tuple for dictionary key
    v_np = -1.0 * pot_drv.compute(molecule, basis, [charge], [position]).to_numpy()
    v_c[position_tuple] = np.einsum("ab, ab ->", D, v_np) #* weight  # Multiply by weight
    #print("v_c[position_tuple]",v_c[position_tuple])
# Print results
for position, value in v_c.items():
    print(value)

