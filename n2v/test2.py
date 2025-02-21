import basis_set_exchange as bse
from basis_set_exchange.convert import convert_formatted_basis_str
import re
import numpy as np


def get_veloxchem_basis(elements, basis_name):
    """
    Extracts a basis set from Basis Set Exchange (BSE) in NWChem format and converts it to VeloxChem format.
    """
    veloxchem_basis = {}

    for element in elements:
        formatted_element = element.capitalize()

        # Retrieve basis set in NWChem format
        basis_nwchem = bse.get_basis(basis_name, elements=[formatted_element], fmt='nwchem')

        # Convert to VeloxChem format
        basis_veloxchem = convert_formatted_basis_str(basis_nwchem, 'nwchem', 'veloxchem')
        print(f"basis_veloxchem for {element}:", basis_veloxchem)

        # Store result
        veloxchem_basis[formatted_element] = basis_veloxchem

    return veloxchem_basis

def parse_basis_set(basis_str):
    """Parse a basis set string into a dictionary-like structure."""
    basis_data = {}

    # Split into different atoms
    atom_data = re.split(r'@ATOMBASIS (\w+)', basis_str)

    for atom_entry in atom_data[1:]:  # Ignore the first empty entry
        # Extract the element symbol
        lines = atom_entry.strip().split('\n')
        element = lines[0].strip()

        # Initialize a dictionary to store basis data for this atom
        element_basis = {
            'electron_shells': []
        }

        current_shell = None

        for line in lines[1:]:
            line = line.strip()  # Strip leading/trailing whitespaces
            if not line:  # Skip empty lines
                continue

            if line.startswith("S") or line.startswith("P"):  # Shell type
                # If there's an existing shell, add it
                if current_shell:
                    element_basis['electron_shells'].append(current_shell)

                # Initialize a new shell
                try:
                    shell_type, n, m = line.split()
                    current_shell = {
                        'type': shell_type,
                        'n': int(n),
                        'm': int(m),
                        'coefficients': [],
                        'angular_momentum': []  # Added angular momentum to match your structure
                    }
                except ValueError:
                    print(f"Error processing shell line: {line}")
                    continue  # Skip problematic lines

            else:
                # Coefficients for the current shell
                if current_shell is not None:
                    try:
                        coefficients = list(map(float, line.split()))
                        current_shell['coefficients'].append(coefficients)

                        # Assuming 'S' or 'P' correspond to angular momentum
                        if current_shell['type'] == 'S':
                            current_shell['angular_momentum'].append(0)  # S orbitals have angular momentum 0
                        elif current_shell['type'] == 'P':
                            current_shell['angular_momentum'].append(1)  # P orbitals have angular momentum 1
                    except ValueError:
                        print(f"Error processing coefficient line: {line}")
                        continue  # Skip problematic lines

        # Append the last shell if it's valid
        if current_shell:
            element_basis['electron_shells'].append(current_shell)

        basis_data[element] = element_basis

    return basis_data



def convert_to_generalized_shell(self):
    generalized_shells = []

    # Get atomic numbers and their coordinates
    elem_ids = self.mol.elem_ids_to_numpy()

    for atom_index, (atomic_number, coord) in enumerate(zip(elem_ids, self.atomic_coords)):
        element_symbol = self.atomic_number_to_symbol[str(atomic_number)]

        if element_symbol not in self.element_basis_data:
            print(f"No basis data found for element {element_symbol}, skipping atom {atom_index}...")
            continue

        # Retrieve the basis set for this atom's element
        element_basis = self.element_basis_data[element_symbol]

        for shell_data in element_basis['electron_shells']:
            # Handle multiple angular momenta per shell
            for l_ang, coeff_list in zip(shell_data['angular_momentum'], shell_data['coefficients']):
                exponents = np.array([float(exp) for exp in shell_data['exponents']], dtype=float)
                coefficients = np.array([float(coeff) for coeff in coeff_list], dtype=float)

                print(f"Processing atom {atom_index}: {element_symbol} at {coord}")
                print(f"Angular Momentum: {l_ang}")
                print("Exponents:", exponents)
                print("Coefficients:", coefficients)

                coord_type = 'cartesian'  # Adjust as needed

                # Create and store the shell for this atom
                converted_shell = GeneralizedContractionShell(l_ang, coord, coefficients, exponents, coord_type)
                generalized_shells.append(converted_shell)

    return generalized_shells


veloxchem_basis = get_veloxchem_basis(["Be"], "6-31g")

# Example usage:
basis_veloxchem = veloxchem_basis
parsed_basis = parse_basis_set(basis_veloxchem)

print(parsed_basis)  # Now, parsed_basis will be a dictionary structure



