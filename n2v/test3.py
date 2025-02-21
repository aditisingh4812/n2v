import json
import basis_set_exchange as bse
from basis_set_exchange.convert import convert_formatted_basis_str
import re
class BasisSetExtractor:
    def __init__(self):
        # Mapping of atomic numbers to element symbols
        self.atomic_number_to_symbol = {
            '1': 'H', '2': 'He', '3': 'Li', '4': 'Be', '5': 'B', '6': 'C', '7': 'N', '8': 'O',
            '9': 'F', '10': 'Ne', '11': 'Na', '12': 'Mg', '13': 'Al', '14': 'Si', '15': 'P', '16': 'S',
            '17': 'Cl', '18': 'Ar', '19': 'K', '20': 'Ca', '21': 'Sc', '22': 'Ti', '23': 'V', '24': 'Cr',
            '25': 'Mn', '26': 'Fe', '27': 'Co', '28': 'Ni', '29': 'Cu', '30': 'Zn', '31': 'Ga', '32': 'Ge',
            '33': 'As', '34': 'Se', '35': 'Br', '36': 'Kr'
        }

    def get_basis_set_for_elements(self, elements, basis_name):
        """
        Extracts a basis set from Basis Set Exchange (BSE) in NWChem format and converts it to VeloxChem format.
        """
        veloxchem_basis = {}
        basis_data = bse.get_basis(basis_name, fmt='json')
        basis_dict = json.loads(basis_data)  # Parse the JSON data

        # Collect basis data for each element
        for element in elements:
            # Find the atomic number corresponding to the element
            atomic_number = [key for key, value in self.atomic_number_to_symbol.items() if value == element][0]
            print(f"atomic_number for {element}: {atomic_number}")

            if atomic_number in basis_dict['elements']:
                element_data = basis_dict['elements'][atomic_number]
                veloxchem_basis[element] = element_data
            else:
                print(f"Basis set not found for element: {element}")

            # Retrieve the basis set in NWChem format for conversion
            basis_nwchem = bse.get_basis(basis_name, elements=[element], fmt='nwchem')

            # Convert to VeloxChem format
            basis_veloxchem = convert_formatted_basis_str(basis_nwchem, 'nwchem', 'veloxchem')
            print(f"Converted VeloxChem basis for {element}: {basis_veloxchem}")

            # Store the converted basis set
            veloxchem_basis[element] = basis_veloxchem

        return basis_veloxchem


# Example usage
extractor = BasisSetExtractor()
elements = ['Ne']  # Specify the elements you want
basis_name = "6-31g"  # Specify the desired basis set

veloxchem_basis = extractor.get_basis_set_for_elements(elements, basis_name)
print("veloxchem_basis",veloxchem_basis)





def extract_basis_data(veloxchem_basis_str):
    # Angular momentum mapping
    angular_momentum_map = {'S': 0, 'P': 1, 'D': 2, 'F': 3}

    # Regular expression to match shell headers like "S 1 1", "P 3 1", etc.
    shell_pattern = re.compile(r"(S|P|D|F)\s+(\d+)\s+\d+")

    # Find all matches for shell types, number of basis functions, and positions
    shell_matches = [(match.group(), match.start()) for match in shell_pattern.finditer(veloxchem_basis_str)]

    # List to preserve order
    basis_data = []

    # Iterate through each match and extract data
    for i, (shell_line, start_pos) in enumerate(shell_matches):
        shell, num_basis_functions, _ = shell_line.split()
        angular_momentum = angular_momentum_map[shell]  # Convert shell to angular momentum number
        num_basis_functions = int(num_basis_functions)
        
        print(f"Processing: {shell_line} (l = {angular_momentum}) at position {start_pos}")

        # Find the next occurrence to determine data end
        end_pos = shell_matches[i + 1][1] if i + 1 < len(shell_matches) else len(veloxchem_basis_str)

        # Extract the corresponding data section
        data_section = veloxchem_basis_str[start_pos:end_pos].strip().split("\n")[1:]  # Skip the header line
        
        # Initialize storage for exponents and coefficients
        exponents = []
        coefficients = []

        for line in data_section:
            parts = line.split()
            if len(parts) < 2:
                continue

            # Append exponents and coefficients
            try:
                exponents.append(float(parts[0]))
                coefficients.append(float(parts[1]))
            except ValueError:
                continue  # Skip any non-numerical values

        # Store in the ordered list
        basis_data.append({
            'angular_momentum': angular_momentum,  # Use the quantum number instead of shell letter
            'num_basis_functions': num_basis_functions,
            'exponents': exponents,
            'coefficients': coefficients
        })

    return basis_data

# Example VeloxChem basis set string for testing
veloxchem_basis_str = """
@BASIS_SET 6-31G

! NEON       (10s,4p) -> [3s,2p]
@ATOMBASIS NE
S    1    1
0.4458187000E+00  0.1000000000E+01
S    3    1
0.2653213100E+02 -0.1071182872E+00
0.6101755010E+01 -0.1461638213E+00
0.1696271530E+01  0.1127773503E+01
P    3    1
0.2653213100E+02  0.7190958851E-01
0.6101755010E+01  0.3495133720E+00
0.1696271530E+01  0.7199405121E+00
P    1    1
0.4458187000E+00  0.1000000000E+01
S    6    1
0.8425851530E+04  0.1884348050E-02
0.1268519400E+04  0.1433689940E-01
0.2896214140E+03  0.7010962331E-01
0.8185900400E+02  0.2373732660E+00
0.2625150790E+02  0.4730071261E+00
0.9094720510E+01  0.3484012410E+00
@END
"""

# Extract the basis data
basis_data = extract_basis_data(veloxchem_basis_str)

# Print the results
print("\nExtracted Basis Data (Preserving Order & Using Angular Momentum Numbers):")
for entry in basis_data:
    print(f"Angular Momentum (l): {entry['angular_momentum']}")
    print(f"  Number of Basis Functions: {entry['num_basis_functions']}")
    print(f"  Exponents: {entry['exponents']}")
    print(f"  Coefficients: {entry['coefficients']}")

exit()

import re

def extract_basis_data(veloxchem_basis_str):
    # Regular expression to match shell headers like "S 1 1", "P 3 1", etc.
    shell_pattern = re.compile(r"(S|P|D|F)\s+(\d+)\s+\d+")

    # Find all the matches for shell types and the number of basis functions with their positions
    shell_matches = [(match.group(), match.start()) for match in shell_pattern.finditer(veloxchem_basis_str)]

    # List to preserve order
    basis_data = []

    # Iterate through each match and extract data
    for i, (shell_line, start_pos) in enumerate(shell_matches):
        shell, num_basis_functions, _ = shell_line.split()
        num_basis_functions = int(num_basis_functions)
        
        print(f"Processing: {shell_line} at position {start_pos}")

        # Find the next occurrence to determine data end
        end_pos = shell_matches[i + 1][1] if i + 1 < len(shell_matches) else len(veloxchem_basis_str)

        # Extract the corresponding data section
        data_section = veloxchem_basis_str[start_pos:end_pos].strip().split("\n")[1:]  # Skip the header line
        
        # Initialize storage for exponents and coefficients
        exponents = []
        coefficients = []

        for line in data_section:
            parts = line.split()
            if len(parts) < 2:
                continue

            # Append exponents and coefficients
            try:
                exponents.append(float(parts[0]))
                coefficients.append(float(parts[1]))
            except ValueError:
                continue  # Skip any non-numerical values

        # Store in the ordered list
        basis_data.append({
            'shell': shell,  # Keep the exact sequence
            'num_basis_functions': num_basis_functions,
            'exponents': exponents,
            'coefficients': coefficients
        })

    return basis_data

# Example VeloxChem basis set string for testing
veloxchem_basis_str = """
@BASIS_SET 6-31G

! NEON       (10s,4p) -> [3s,2p]
@ATOMBASIS NE
S    1    1
0.4458187000E+00  0.1000000000E+01
S    3    1
0.2653213100E+02 -0.1071182872E+00
0.6101755010E+01 -0.1461638213E+00
0.1696271530E+01  0.1127773503E+01
P    3    1
0.2653213100E+02  0.7190958851E-01
0.6101755010E+01  0.3495133720E+00
0.1696271530E+01  0.7199405121E+00
P    1    1
0.4458187000E+00  0.1000000000E+01
S    6    1
0.8425851530E+04  0.1884348050E-02
0.1268519400E+04  0.1433689940E-01
0.2896214140E+03  0.7010962331E-01
0.8185900400E+02  0.2373732660E+00
0.2625150790E+02  0.4730071261E+00
0.9094720510E+01  0.3484012410E+00
@END
"""


# Extract the basis data
basis_data = extract_basis_data(veloxchem_basis_str)

# Print the results
print("\nExtracted Basis Data (Preserving Order):")
for entry in basis_data:
    print(f"Shell: {entry['shell']}")
    print(f"  Number of Basis Functions: {entry['num_basis_functions']}")
    print(f"  Exponents: {entry['exponents']}")
    print(f"  Coefficients: {entry['coefficients']}")

exit()
import re

def extract_basis_data(veloxchem_basis_str):
    # Define the angular momentum mapping
    angular_momentum_map = {
        'S': 0,
        'P': 1,
        'D': 2,
        'F': 3
    }

    # Regular expression to match shell headers like "S 1 1", "P 3 1", etc.
    shell_pattern = re.compile(r"(S|P|D|F)\s+(\d+)\s+\d+")

    # Find all the matches for shell types and the number of basis functions with their positions
    shell_matches = [(match.group(), match.start()) for match in shell_pattern.finditer(veloxchem_basis_str)]

    basis_data = {}

    # Iterate through each match and extract data
    for i, (shell_line, start_pos) in enumerate(shell_matches):
        shell, num_basis_functions, _ = shell_line.split()
        angular_momentum = angular_momentum_map[shell]
        num_basis_functions = int(num_basis_functions)

        print(f"Processing: {shell_line} at position {start_pos}")

        # Find the next occurrence to determine data end
        end_pos = shell_matches[i + 1][1] if i + 1 < len(shell_matches) else len(veloxchem_basis_str)

        # Extract the corresponding data section
        data_section = veloxchem_basis_str[start_pos:end_pos].strip().split("\n")[1:]  # Skip the header line

        # Initialize storage for exponents and coefficients
        exponents = []
        coefficients = []

        for line in data_section:
            parts = line.split()
            if len(parts) < 2:
                continue

            # Append exponents and coefficients
            try:
                exponents.append(float(parts[0]))
                coefficients.append(float(parts[1]))
            except ValueError:
                continue  # Skip any non-numerical values

        # Store in the basis_data dictionary
        if angular_momentum not in basis_data:
            basis_data[angular_momentum] = []

        basis_data[angular_momentum].append({
            'num_basis_functions': num_basis_functions,
            'exponents': exponents,
            'coefficients': coefficients
        })

    return basis_data

# Example VeloxChem basis set string for testing
veloxchem_basis_str = """
@BASIS_SET 6-31G

! NEON       (10s,4p) -> [3s,2p]
@ATOMBASIS NE
S    1    1
0.4458187000E+00  0.1000000000E+01
S    3    1
0.2653213100E+02 -0.1071182872E+00
0.6101755010E+01 -0.1461638213E+00
0.1696271530E+01  0.1127773503E+01
P    3    1
0.2653213100E+02  0.7190958851E-01
0.6101755010E+01  0.3495133720E+00
0.1696271530E+01  0.7199405121E+00
P    1    1
0.4458187000E+00  0.1000000000E+01
S    6    1
0.8425851530E+04  0.1884348050E-02
0.1268519400E+04  0.1433689940E-01
0.2896214140E+03  0.7010962331E-01
0.8185900400E+02  0.2373732660E+00
0.2625150790E+02  0.4730071261E+00
0.9094720510E+01  0.3484012410E+00
@END
"""

# Extract the basis data
basis_data = extract_basis_data(veloxchem_basis_str)

# Print the results
print("\nExtracted Basis Data:")
for angular_momentum, shells in basis_data.items():
    print(f"Angular Momentum {angular_momentum}:")
    for shell in shells:
        print(f"  Number of Basis Functions: {shell['num_basis_functions']}")
        print(f"  Exponents: {shell['exponents']}")
        print(f"  Coefficients: {shell['coefficients']}")


