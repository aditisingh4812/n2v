import veloxchem as vlx
import numpy as np

mol_string = """
O   0.0   0.0   0.0
H   0.0   1.4   1.1
H   0.0  -1.4   1.1
"""
basis_label = 'def2-svp'

mol = vlx.Molecule.read_molecule_string(mol_string, units='au')
bas = vlx.MolecularBasis.read(mol, basis_label)

nbfs = 0

for bf in bas.basis_functions():
    print(bf.get_angular_momentum())

    nbfs += bf.get_angular_momentum() * 2 + 1

    exponents = bf.get_exponents()
    norm_factors = bf.get_normalization_factors()

    for i in range(bf.number_of_primitives()):
        print('  ', exponents[i], norm_factors[i])

print('nbfs=', nbfs)

