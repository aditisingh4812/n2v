import basis_set_exchange as bse
from basis_set_exchange.convert import convert_formatted_basis_str
import re

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
        print("basis_veloxchem", basis_veloxchem)

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
            if line.startswith("S") or line.startswith("P"):  # Shell type
                if current_shell:
                    element_basis['electron_shells'].append(current_shell)

                # New shell, initialize
                shell_type, n, m = line.split()
                current_shell = {
                    'type': shell_type,
                    'n': int(n),
                    'm': int(m),
                    'coefficients': []
                }
            else:
                # Coefficients for the current shell
                current_shell['coefficients'].append(list(map(float, line.split())))

        # Append the last shell
        if current_shell:
            element_basis['electron_shells'].append(current_shell)

        basis_data[element] = element_basis

    return basis_data

# Example usage
veloxchem_basis = get_veloxchem_basis(["Be"], "6-31g")
print("veloxchem_basis",type(veloxchem_basis))
exit()
# Example usage:
basis_veloxchem = veloxchem_basis["Be"]
parsed_basis = parse_basis_set(basis_veloxchem)

print(parsed_basis)  # Now, parsed_basis will be a dictionary structure

# Save to file
with open("veloxchem_basis.txt", "w") as f:
    for element, data in veloxchem_basis.items():
        f.write(f"@ATOMBASIS {element}\n")
        # Write out the parsed basis set information
        for shell in data['electron_shells']:
            f.write(f"{shell['type']} {shell['n']} {shell['m']}\n")
            for coeff in shell['coefficients']:
                f.write(" ".join(map(str, coeff)) + "\n")
        f.write("\n@END\n")

print("VeloxChem basis set extracted successfully.")

