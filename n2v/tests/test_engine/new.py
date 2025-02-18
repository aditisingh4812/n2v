import numpy as np
import veloxchem as vlx
h2o_xyz = """1

Ne        0.00000000    0.00000000    0.00000
"""

molecule = vlx.Molecule.from_xyz_string(h2o_xyz)
basis = vlx.MolecularBasis.read(molecule, "sto-3g")
scf_drv = vlx.ScfRestrictedDriver()
scf_drv.ostream.mute()

scf_results = scf_drv.compute(molecule, basis)

D = scf_results["D_alpha"] + scf_results["D_beta"]
positions = np.load('all.npy')#[(0.0, 0.0, 0.0), (0.0, 0.0, 5.0)]  # replace with grid points
print("positions",positions)

charges = -1.0 * np.ones(len(positions))
print("charges",charges)
#print(dir(vlx))
#T = vlx.compute_kinetic_energy_integrals(molecule, basis)
#pot_drv = vlx.compute_nuclear_potential_integrals(molecule, basis)

pot_drv = vlx.NuclearPotentialIntegralsDriver()         #vlx.NuclearPotentialDriver()
v_c = {}
for charge, position in zip(charges, positions):
    v_np = -1.0 * pot_drv.compute(molecule, basis, [charge], [position]).to_numpy()
    v_c[position] = np.einsum("ab, ab ->", D, v_np)
for position in v_c.keys():
    print(position, v_c[position])
