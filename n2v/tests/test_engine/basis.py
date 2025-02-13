import veloxchem as vlx

molecule_data = """2

H    0.000000000000        0.740848095288        0.582094932012
H    0.000000000000       -0.740848095288        0.582094932012
"""

# Instead of passing a Molecule object, you pass the molecule_data string
molecule = vlx.Molecule.read_xyz_string(molecule_data)

# Load the molecule and the BSE basis set
#basis = vlx.MolecularBasis.read_bse('/home/aditisingh4812/velox_chem/n2v/n2v/tests/test_engine/3-21.vlx')

basis = vlx.MolecularBasis.read(molecule, "sto-3g", ostream=None)
print(dir(basis))


# Specify the angular momentum (e.g., 0 for s orbitals, 1 for p orbitals, etc.)
angular_momentum = 0  # Example for s orbitals

# Now call n_basis_functions with the molecule and angular momentum
num_basis_functions = basis.n_basis_functions(molecule, angular_momentum)
print(f"Number of basis functions (with angular momentum {angular_momentum}): {num_basis_functions}")


# Get available basis sets
avail_basis = basis.get_avail_basis()
print(f"Available basis functions: {avail_basis}")

# Specify angular momentum (e.g., 0 for s orbitals, 1 for p orbitals, etc.)
angular_momentum = 0  # Example: s orbitals

# Get number of basis functions for the specified angular momentum
num_basis_functions = basis.n_basis_functions(molecule, angular_momentum)
print(f"Number of basis functions (angular momentum {angular_momentum}): {num_basis_functions}")

# Get atomic orbital to basis function map
ao_basis_map = basis.get_ao_basis_map()
print(f"Atomic orbital to basis function map: {ao_basis_map}")

exit()
# Extract shells, contractions, and atomic contributions
shells = basis.get_shells()
contractions = basis.get_contractions()
atomic_contributions = basis.get_atomic_contributions()

# Print shell information
for shell in shells:
    print(f"Shell: {shell}")

# Print contraction information
for contraction in contractions:
    print(f"Contraction: {contraction}")

# Print atomic contributions
for atom, contrib in atomic_contributions.items():
    print(f"Atom: {atom}, Contribution: {contrib}")

