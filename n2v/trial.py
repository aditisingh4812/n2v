import basis_set_exchange as bse
import json
import csv

basis_name = 'cc-pVTZ'  # Example basis set

# Download basis set data in JSON format
basis_data = bse.get_basis(basis_name, fmt='json')
basis_dict = json.loads(basis_data)

# Extract relevant data
elements = basis_dict['elements']

# Open a CSV file to store the extracted data
with open(f'{basis_name}_details.csv', 'w', newline='') as csvfile:
    fieldnames = ['Element', 'Shell Angular Momentum', 'Exponents', 'Coefficients']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for element, data in elements.items():
        for shell in data['electron_shells']:
            # Extract angular momentum, exponents, and coefficients
            ang_mom = shell['angular_momentum']
            exponents = shell['exponents']
            coefficients = shell['coefficients']

            writer.writerow({
                'Element': element,
                'Shell Angular Momentum': ', '.join(map(str, ang_mom)),
                'Exponents': ', '.join(exponents),
                'Coefficients': ', '.join(str(c) for c in coefficients[0])  # Assuming one set of coefficients
            })

print(f"Basis set details with angular momentum saved to {basis_name}_details.csv")

