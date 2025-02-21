import veloxchem as vlx

# Define molecule
molecule_data = """2

Ne        0.00000000    0.00000000    0.00000
He        0.0           0.0           1.3
"""

# Read molecule and assign basis set
molecule = vlx.Molecule.read_xyz_string(molecule_data)
basis_name = '6-31G'
basis1 = vlx.MolecularBasis.read(molecule, basis_name, ostream=None)

# Get basis set information
print("\n=== Basis Set Information ===")
print(f"Basis Set: {basis_name}")  # Use the manually assigned name

# Access the basis set data via VeloxChem's internal string representation
basis_str = str(basis1)
print("Basis Set Details:")
print(basis_str)

# Extract data using string parsing (example for He and Ne)
import re

# Split the basis string into atomic blocks
atomic_blocks = re.split(r'_atomicBasisSets\[\d+\]:', basis_str)

for block in atomic_blocks[1:]:  # Skip the header
    # Extract elemental ID
    z_match = re.search(r'_idElemental: (\d+)', block)
    if z_match:
        z = int(z_match.group(1))
        element = vlx.ElementalData().get_element_symbol(z)
        print(f"\nElement: {element} (Z={z})")
    
    # Extract basis functions
    basis_funcs = re.findall(
        r'\[CBasisFunction.*?_angularMomentum: (\d+)(.*?)_exponents:(.*?)_normFactors:(.*?)\]', 
        block, 
        re.DOTALL
    )
    
    for idx, (ang_mom, _, exponents, norms) in enumerate(basis_funcs):
        ang_mom = int(ang_mom)
        exponents = re.findall(r'\d+\.\d+', exponents)
        norms = re.findall(r'-?\d+\.\d+', norms)
        
        orbital = 'spdf'[ang_mom] if ang_mom < 4 else f"l={ang_mom}"
        print(f"  Basis Function {idx+1}: {orbital}-orbital")
        print(f"    Exponents: {exponents}")
        print(f"    Norms: {norms}\n")
