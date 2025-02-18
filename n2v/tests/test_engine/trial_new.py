import veloxchem as vlx

# Function to read the VeloxChem basis set file
def read_veloxchem_basis(basis_file):
    try:
        with open(basis_file, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: The file {basis_file} was not found.")
        return []
    
    basis_data = []
    current_shell = None

    # Parsing the file
    for line in lines:
        line = line.strip()
        
        if line.startswith('@ATOMBASIS'):
            # Only extract shell information, not atom symbol
            current_shell = line.split()[1]
            print(f"Shell identified: {current_shell}")  # Debugging the shell info
        elif line and not line.startswith(('!', '@')):
            # Reading data for each basis function
            parts = line.split()
            if len(parts) == 3:
                try:
                    # Shell type (e.g., 'S', 'P', 'D') corresponds to the angular momentum (l)
                    shell_type = parts[0].upper()
                    exponent = float(parts[1])
                    coeff = float(parts[2])

                    # Map shell types to angular momentum (l): 'S' -> 0, 'P' -> 1, 'D' -> 2
                    if shell_type == 'S':
                        l = 0
                    elif shell_type == 'P':
                        l = 1
                    elif shell_type == 'D':
                        l = 2
                    else:
                        print(f"Warning: Unknown shell type '{shell_type}' in line: {line}")
                        continue
                    
                    # Store data in basis_data
                    basis_data.append((current_shell, exponent, l, coeff))
                except ValueError:
                    print(f"Warning: Invalid line format: {line}")
    return basis_data

# Function to create generalized contractions
def create_generalized_contraction(basis_data):
    contractions = {}
    for shell, exponent, l, coeff in basis_data:
        if shell not in contractions:
            contractions[shell] = []
        contractions[shell].append((l, exponent, coeff))
    return contractions

# Function to add the generalized basis set to a VeloxChem molecule
def add_generalized_basis_to_veloxchem(molecule, contractions):
    mol_basis = vlx.MolecularBasis()
    
    # Use a method that is compatible with VeloxChem to add the basis set
    for shell, primitives in contractions.items():
        print(f"Adding shell: {shell} with primitives {primitives}")  # Debugging
        try:
            mol_basis.add(molecule, shell, primitives)
        except AttributeError:
            print(f"Error: Failed to add shell {shell} with primitives {primitives}")
            continue
    return mol_basis

# Main execution
basis_file = 'ano_vt_tz.veloxchem'
basis_data = read_veloxchem_basis(basis_file)

if basis_data:
    contractions = create_generalized_contraction(basis_data)

    # Create a sample molecule (Helium atom)
    molecule = vlx.Molecule.read_xyz_string("""1
    He atom
    He 0.0 0.0 0.0
    """)

    # Add the generalized contracted basis set
    mol_basis = add_generalized_basis_to_veloxchem(molecule, contractions)
    
    print("Generalized contracted basis set successfully added to VeloxChem")
else:
    print("Error: No valid basis data was found.")

