"""
grider_veloxchem.py
Grider for VeloxChem
"""
import inspect
import re
import veloxchem 
from gbasis.evals.density import (
    evaluate_density,
    evaluate_density_laplacian,
    evaluate_density_gradient)
from gbasis.evals.density import evaluate_density
from gbasis.evals.eval import evaluate_basis
from gbasis.evals.eval_deriv import evaluate_deriv_basis
from gbasis.evals.electrostatic_potential import point_charge_integral
from gbasis.integrals.electron_repulsion import electron_repulsion_integral
from gbasis.contractions import GeneralizedContractionShell

import numpy as np
from opt_einsum import contract
import basis_set_exchange as bse
import json
from basis_set_exchange.convert import convert_formatted_basis_str

print(inspect.getfile(point_charge_integral))
try:
    import veloxchem
    has_veloxchem = True
except ImportError:
    has_veloxchem = False
if has_veloxchem:
    import veloxchem as vlx
    from veloxchem import GridDriver
    import numpy as np
    from gbasis.evals.eval import evaluate_basis
    from gbasis.evals.electrostatic_potential import point_charge_integral

    class VeloxchemGrider:
        def __init__(self, mol, pbs_mol=None, basis_str=None, ref=None, grid_level=1,scf_results=None):
           self.mol = mol
        # Handle user-defined basis set
           self.basis = vlx.MolecularBasis.read(mol, basis_str)
           print("self basis shape",dir(self.basis))
           self.pbs = vlx.MolecularBasis.read(pbs_mol,basis_str) if pbs_mol else None
           # Define the basis set name you want to use
           self.basis_name = basis_str
           # Extract element IDs
           elem_ids = mol.elem_ids_to_numpy()
           print("elem_ids",elem_ids)
           self.atomic_number_to_symbol = {
           '1': 'H', '2': 'He', '3': 'Li', '4': 'Be', '5': 'B', '6': 'C', '7': 'N', '8': 'O',
           '9': 'F', '10': 'Ne', '11': 'Na', '12': 'Mg', '13': 'Al', '14': 'Si', '15': 'P', '16': 'S',
           '17': 'Cl', '18': 'Ar', '19': 'K', '20': 'Ca', '21': 'Sc', '22': 'Ti', '23': 'V', '24': 'Cr',
           '25': 'Mn', '26': 'Fe', '27': 'Co', '28': 'Ni', '29': 'Cu', '30': 'Zn', '31': 'Ga', '32': 'Ge',
           '33': 'As', '34': 'Se', '35': 'Br', '36': 'Kr'
            }
           # Map atomic numbers to element symbols
           unique_elements = list([self.atomic_number_to_symbol[str(atom)] for atom in elem_ids])
           print("unique_elements",unique_elements)
           # Extract the basis set from Basis Set Exchange (BSE)
           #self.basis_exchange = self.get_basis_set_for_elements(unique_elements,basis_str)
           self.element_basis_data = self.get_basis_set_for_elements(unique_elements,basis_str)
           #self.extract_basis_data = self.extract_basis_data()
           self.ref = ref
           try:
            #self.atomic_charges = np.array(self.mol.get_charge())
            atomic_composition = self.mol.get_elemental_composition()
            atomic_charges = np.array([])
            for element in atomic_composition:
                 atomic_charges = np.append(atomic_charges,element)
            self.atomic_charges = atomic_charges
            print(f"atomic_charges shape: {self.atomic_charges.shape}")# Replace with the correct method if necessary
           except AttributeError:
            print("Error: 'get_nuclear_charges' not found in the Molecule class.")
           try:
            # Assuming that you want to get the coordinates for all atoms
            self.atomic_coords = np.array([self.mol.get_atom_coordinates(i) for i in range(self.mol.number_of_atoms())])
            print(f"atomic_coords shape: {self.atomic_coords.shape}")
            print("self.atomic_coords",self.atomic_coords)
            print(type(self.atomic_coords))  # Should be <class 'numpy.ndarray'>

           except AttributeError:
            print("Error: 'get_nuclear_coordinates' not found in the Molecule class.")
           # Perform SCF Calculation based on ref
           if self.ref == 1:
             scf_drv = vlx.ScfRestrictedDriver()
           else:
             scf_drv = vlx.ScfUnrestrictedDriver()
           scf_drv.xcfun = "slater"
           self.grid_level = grid_level
           grid_drv = GridDriver()
           grid_drv.set_level(grid_level)
           molgrid = grid_drv.generate(mol) # Generate grid for the molecule
           self.molgrid = molgrid
           print("molgrid_dir", dir(molgrid))
           # Step 2: Access grid points and weights
           x_coords = molgrid.x_to_numpy()  # Get x coordinates as a NumPy array
           y_coords = molgrid.y_to_numpy()  # Get y coordinates as a NumPy array
           z_coords = molgrid.z_to_numpy()  # Get z coordinates as a NumPy array
           coords = np.vstack((x_coords, y_coords, z_coords)).T  # Combine into a single array of coordinates
           print("coords.shape",coords.shape)
           self.spherical_points = coords  # Get grid points as NumPy array
           self.w = molgrid.w_to_numpy()  # Get weights for integration
           self.rectangular_grid = None
           if scf_results is not None:
              self.scf_results = scf_results
           else: 
              self.scf_results = scf_drv.compute(self.mol, self.basis)

           # Extract density matrix from SCF results
           self.Da = self.scf_results['D_alpha']
           if self.ref != 1:
            self.Db = self.scf_results['D_beta']
           self.converted = self.convert_to_generalized_shell()
        # Function to extract basis set details from BSE for a given element
        
        def get_basis_set_for_elements(self, elements, basis_name):
            atomic_number_to_symbol = {
                '1': 'H', '2': 'He', '3': 'Li', '4': 'Be', '5': 'B', '6': 'C', '7': 'N', '8': 'O',
                '9': 'F', '10': 'Ne', '11': 'Na', '12': 'Mg', '13': 'Al', '14': 'Si', '15': 'P', '16': 'S',
                '17': 'Cl', '18': 'Ar', '19': 'K', '20': 'Ca', '21': 'Sc', '22': 'Ti', '23': 'V', '24': 'Cr',
                '25': 'Mn', '26': 'Fe', '27': 'Co', '28': 'Ni', '29': 'Cu', '30': 'Zn', '31': 'Ga', '32': 'Ge',
                '33': 'As', '34': 'Se', '35': 'Br', '36': 'Kr'
            }

            basis_data = bse.get_basis(basis_name, fmt='json')
            basis_dict = json.loads(basis_data)

            # Print available elements in the basis set
            print(f"Available elements in {basis_name}:")
            print(basis_dict['elements'].keys())

            element_basis_data = {}

            for element in elements:
                # Get atomic number from the element symbol
                atomic_number = [key for key, value in atomic_number_to_symbol.items() if value == element][0]
                print("atomic_number",atomic_number)
                if atomic_number in basis_dict['elements']:
                    element_data = basis_dict['elements'][atomic_number]
                    element_basis_data[element] = element_data
                else:
                    print(f"Basis set not found for element: {element}")

            print("element_basis_data",element_basis_data)
            return element_basis_data 
        '''
        def get_basis_set_for_elements(self,elements, basis_name):
            """
            Extracts a basis set from Basis Set Exchange (BSE) in NWChem format and converts it to VeloxChem format.
            """
            atomic_number_to_symbol = {
                '1': 'H', '2': 'He', '3': 'Li', '4': 'Be', '5': 'B', '6': 'C', '7': 'N', '8': 'O',
                '9': 'F', '10': 'Ne', '11': 'Na', '12': 'Mg', '13': 'Al', '14': 'Si', '15': 'P', '16': 'S',
                '17': 'Cl', '18': 'Ar', '19': 'K', '20': 'Ca', '21': 'Sc', '22': 'Ti', '23': 'V', '24': 'Cr',
                '25': 'Mn', '26': 'Fe', '27': 'Co', '28': 'Ni', '29': 'Cu', '30': 'Zn', '31': 'Ga', '32': 'Ge',
                '33': 'As', '34': 'Se', '35': 'Br', '36': 'Kr'
            }

            veloxchem_basis = {}
            basis_data = bse.get_basis(basis_name, fmt='json')
            basis_dict = json.loads(basis_data)
            element_basis_data = {}
            for element in elements:
                atomic_number = [key for key, value in self.atomic_number_to_symbol.items() if value == element][0]
                print("atomic_number",atomic_number)
                formatted_element = element.capitalize()
                print("formatted_element",formatted_element)
                if atomic_number in basis_dict['elements']:
                    element_data = basis_dict['elements'][atomic_number]
                    element_basis_data[element] = element_data
                else:
                    print(f"Basis set not found for element: {element}")
                # Retrieve basis set in NWChem format
                basis_nwchem = bse.get_basis(basis_name, elements=[element], fmt='nwchem')
                print("basis_nwchem",basis_nwchem)
                # Convert to VeloxChem format
                basis_veloxchem = convert_formatted_basis_str(basis_nwchem, 'nwchem', 'veloxchem')
            print("basis_veloxchem",basis_veloxchem) 
            return basis_veloxchem
         
        def extract_basis_data(self):
            veloxchem_basis_str = self.element_basis_data
            print("veloxchem_basis_str",veloxchem_basis_str)
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
            print("basis_data",basis_data)
            return basis_data
        
        def convert_to_generalized_shell(self):
            generalized_shells = []
            # Get atomic numbers and their coordinates
            elem_ids = self.mol.elem_ids_to_numpy()
            print("self.extract_basis_data",self.extract_basis_data)
            for atom_index, (atomic_number, coord) in enumerate(zip(elem_ids, self.atomic_coords)):
                element_symbol = self.atomic_number_to_symbol[str(atomic_number)]
                print("element_symbol",element_symbol)
                element_basis = self.extract_basis_data  # Since it's already the correct data

                for shell_data in element_basis:
                     l_ang = shell_data['angular_momentum']
                     exponents = np.array(shell_data['exponents'], dtype=float)
                     coefficients = np.array(shell_data['coefficients'], dtype=float)
                 
                     print(f"Processing atom {atom_index}: {element_symbol} at {self.atomic_coords}")
                     print(f"Angular Momentum: {l_ang}")
                     print("Exponents:", exponents)
                     print("Coefficients:", coefficients)
                 
                     coord_type = 'cartesian'  # Adjust as needed
                     converted_shell = GeneralizedContractionShell(l_ang, coord, coefficients, exponents, coord_type)
                     generalized_shells.append(converted_shell)

            return generalized_shells
        '''
        def convert_to_generalized_shell(self):
            generalized_shells = []
            # Get atomic numbers and their coordinates
            elem_ids = self.mol.elem_ids_to_numpy()
            for atom_index, (atomic_number, coord) in enumerate(zip(elem_ids, self.atomic_coords)):
                element_symbol = self.atomic_number_to_symbol[str(atomic_number)]
                print("element_symbol",element_symbol)
                if element_symbol not in self.element_basis_data:
                    print(f"No basis data found for element {element_symbol}, skipping atom {atom_index}...")
                    continue
                # Retrieve the basis set for this atom's element
                element_basis = self.element_basis_data[element_symbol]
                #print("element_basis",element_basis)
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
        def assert_grid(self, grid):
            if grid == 'spherical':
                points = self.spherical_points
                print("points_shape",points.shape)
            elif grid == 'rectangular':
                assert self.rectangular_grid is not None, "Rectangular Grid must be defined first"
                points = self.rectangular_grid
                print("points_shape",points.shape)

            else:
                raise ValueError("Specify either spherical or rectangular grid")

            return points
        def generate_grid(self, x, y, z):
            """
            Genrates Mesh from 3 separate linear spaces and flatten,
            needed for cubic grid.
            Parameters
            ----------
            grid: tuple of three np.ndarray
                (x, y, z)
            Returns
            -------
            grid: np.ndarray
                shape (3, len(x)*len(y)*len(z)).
            """
            # x,y,z, = grid
            shape = (len(x), len(y), len(z))
            X,Y,Z = np.meshgrid(x, y, z, indexing='ij')
            X = X.reshape((X.shape[0] * X.shape[1] * X.shape[2], 1))
            Y = Y.reshape((Y.shape[0] * Y.shape[1] * Y.shape[2], 1))
            Z = Z.reshape((Z.shape[0] * Z.shape[1] * Z.shape[2], 1))
            grid = np.concatenate((X,Y,Z), axis=1).T

            return grid, shape

        def build_rectangular(self, npoints):
            """
            Builds a rectangular grid that encompasses the molecule.
            Parameters
            ----------
            npoints: tuple
            Number of points per dimension (n_x, n_y, n_z)
            overage: float
            Spatial extent to extend the grid around the molecule (default: 3.0 Å)
            """
            g1 = np.linspace(-10, 10, npoints[0])
            g2 = np.linspace(0, 0, npoints[1])
            g3 = np.linspace(0, 0, npoints[2])
            gx, gy, gz = np.meshgrid(g1, g2, g3)
            g3d = np.vstack( [gx.ravel(), gy.ravel(), gz.ravel()] ).T

            self.x                = g1
            self.y                = g2
            self.z                = g3
            self.rectangular_grid = g3d


        def density(self, Da, Db=None, grid='spherical'):
            """
            Computes density on grid. 

            Parameters
            ----------

            density: np.ndarray.
                Density in AO basis

            grid: str.
                Type of grid used. Default spherical 
                If 'rectangular' used self.rectangular_grid != None 

            Returns
            -------
            density_g: np.ndarray
                Density on the requested grid    
            """
            points = self.assert_grid(grid)

            density_a = evaluate_density(Da, self.converted, points)
            if Db is not None:
                density_b = evaluate_density(Db, self.converted, points)
                density_g = np.concatenate([density_a, density_b])
                return density_g
            else:
                return density_a

        def hartree(self, density, grid='spherical'):
            """
            Computes Hartree Potential on grid. 

            Parameters
            ----------

            density: np.ndarray.
                Density in AO basis

            grid: str.
                Type of grid used. Default spherical 
                If 'rectangular' used self.rectangular_grid != None 


            Returns
            -------

            hartree_potential: np.ndarray
                Hartree potential on the requested grid
            """        
            points = self.assert_grid(grid)

            hartree_potential = point_charge_integral(self.converted, 
                                                    points, 
                                                    -np.ones(points.shape[0]), 
                                                    transform=None)

            hartree_potential *= density[:, :, None]
            hartree_potential = np.sum(hartree_potential, axis=(0, 1))

            return hartree_potential

        def external(self, grid='spherical'):
            """
            Computes External Potential on grid. 

            Parameters
            ----------
            grid: str
                Type of grid used. Default spherical 
                If 'rectangular' used self.rectangular_grid != None

            Returns
            -------
            external_potential: np.ndarray
                External potential on the given grid. 
            """        
            points = self.assert_grid(grid)       

            old_settings = np.seterr(divide="ignore")  # silence warning for dividing by zero
            external_potential = self.atomic_charges[None, :] \
            / (np.sum((points[:, :, None] - self.atomic_coords.T[None, :, :]) ** 2, axis=1) ** 0.5)
            np.seterr(**old_settings)

            if external_potential.ndim > 1:
                external_potential = np.sum(external_potential, axis=1)

            return -external_potential

        def to_grid(self, f_nm, grid='spherical'):
            """
            Expresses a matrix quantity on the grid

            Parameters
            ----------
            coeff: np.ndarray
                Vector/Matrix on ao basis. 
                Shape: {(num_ao_basis, ), (num_ao_basis, num_ao_basis)}
            grid: str
                Type of grid used. Default spherical 
                If 'rectangular' used self.rectangular_grid != None

            Returns
            -------
            f_g: np.ndarray
                Vector/Matrix expressed on the requested grid
            """
            
            points = self.assert_grid(grid)

            if self.pbs is None:
                basis = self.converted
      
            phis = evaluate_basis(basis, points)
            print("phis",phis)
            f_g = f_nm.dot(phis)
            if f_nm.ndim == 2:
                f_g *= phis

            return f_g

        def to_ao(self, f_g, grid='spherical'):
            """
            Expresses grid quantity on the AO basis

            Parameters
            ----------
            f_g: np.ndarray
                Function expressed in g points 
            grid: str
                The grid used: 'radial' or 'spherical'

            Returns
            -------
            f_nm: np.ndarray
                f_g in ao basis
            """

            points = self.assert_grid(grid)

            phis = evaluate_basis(self.converted, points)
            f_nm = contract( 'pb, p,p,pa->ab', phis.T, f_g, self.w, phis.T )
            f_nm = 0.5 * (f_nm + f_nm.T)

            return f_nm
        
        def orbitals(self, C, grid='spherical'):
            """
            Obtains orbitals on grid

            Parameters
            ----------
            C: np.ndarray
                Molecular Orbitals on Atomic Orbital basis set. 
            grid: str
                The grid used: 'radial' or 'spherical'

            Returns
            -------
            mat_g: np.ndarray
                Orbitals in g points in space
            """

            points = self.assert_grid(grid)

            phis = evaluate_basis(self.converted, points)
            mat_g = C.T.dot(phis)
            return mat_g

        def laplacian_density(self, density, grid='spherical'):
            """
            Calculates the laplacian of the density
            
            Parameters
            ----------
            density: np.ndarray
                Density in the ao basis 
            grid: str
                The grid used: 'radial' or 'spherical'

            Returns
            -------
            lap_density: np.ndarray
                Laplacian of density given on g points in space
            """

            points = self.assert_grid(grid)

            lap_density = evaluate_density_laplacian( density, self.converted, points )
            return lap_density
    
        def gradient_density(self, density, grid='spherical'):        
            """
            Evaluatges gradient of density on requested grid

            Parameters
            ----------
            density: np.ndarray
                Density in the ao basis 
            grid: str
                The grid used: 'radial' or 'spherical'

            Returns
            -------
            grad_density: np.ndarray
                Gradient of density given on g points in space
            """

            points = self.assert_grid(grid)

            grad_density = evaluate_density_gradient(density, self.converted, points)
            return grad_density
            
        def ao_deriv(self, derivs=[0,0,0], transform=None, grid='spherical'):
            """ 
            Calculates AO on the grid (and its derivatives). 
            If Transformation is given, e.g. ao2mo, MO will be given

            Parameters
            ----------
            derivs: List of 3 integers
                Array that corresponds to the derivative of each spatial coordinate (x,y,z)
                0 -> no derivative
                1 -> first derivative ...
                [1,0,3] -> first derivative on x. 
                           no derivative on y.
                           third derivative on z

            transform: np.ndarray
                Matrix to transform within basis. E.g. ao2mo -> MO will be given.
            grid: str
                The grid used: 'radial' or 'spherical'

            Returns
            -------
            orbs_deriv: np.ndarray 
                Array of atomic orbitals and/or their derivatives. 
            """

            points = self.assert_grid(grid)

            orbs_deriv = evaluate_deriv_basis( self.converted, points, np.array(derivs), 
                                            transform=transform )

            return orbs_deriv

        # Specialized for methods. 
        # def posdef_kinetic_energy_density(self, density, grid='spherical'):
        #     """
        #     Please look at OuCarter or mRKS method
        #     Evaluates the positive-definite kinetic energy density on grid
        #     t = 1/2 \nabla \cdot \nabla \gamma(r,r') 
        #     """

        #     points = self.assert_grid(grid)
        #     t = evaluate_posdef_kinetic_energy_density(density, self.basis, points)

        #     return t

        # def kinetic_energy_density(self, density, alpha=-1/4, grid='spherical'):
        #     """
        #     Please look at OuCarter or mRKS method
        #     Evaluates the general form of the kinetic energy density
        #     t = 1/2 \nabla \cdot \nabla \gamma(r,r') + alpha \nabla^2 n(r)
        #     """

        #     points = self.assert_grid(grid)
        #     t = evaluate_general_kinetic_energy_density(density, self.basis, points, alpha=alpha)

        #     return t

        # def kinetic_energy_density_pauli(self, C, grid='spherical', method='grid'):
        #     """
        #     Please look at OuCarter or mRKS method
        #     Obtains kinetic energy density in terms of the Pauli kinetic energy density
            
        #     Parameters
        #     ----------
        #     C: np.ndarray
        #         Occupied Molecular Orbitals
        #     """

        #     points = self.assert_grid(grid)

        #     density = C @ C.T
        #     density_g = self.density(density, grid=grid)

        #     basis_dx = self.ao_deriv(derivs=[1,0,0], transform=None, grid=grid)
        #     basis_dy = self.ao_deriv(derivs=[0,1,0], transform=None, grid=grid)
        #     basis_dz = self.ao_deriv(derivs=[0,0,1], transform=None, grid=grid)

        #     if method == 'grid':
        #         orbs = self.orbitals(C, grid=grid)
        #         d_orbs = ((basis_dx + basis_dy + basis_dz).T @ C).T
        #         tau_p = np.zeros_like( density_g )
        #         for i in range(C.shape[1]):
        #             for j in range(C.shape[1]):
        #                 if i == j:
        #                     pass
        #                 else:
        #                     tau_p += np.abs( orbs[i,:] * (d_orbs[j,:]) - orbs[j,:] * (d_orbs[i,:]) )**2

        #     elif method == 'basis':
        #         basis = self.ao_deriv(grid=grid)
        #         dx = contract('pm,mi,nj,pn->ijp', basis.T, C, C, basis_dx.T)
        #         dy = contract('pm,mi,nj,pn->ijp', basis.T, C, C, basis_dy.T)
        #         dz = contract('pm,mi,nj,pn->ijp', basis.T, C, C, basis_dz.T)
            
        #         dx = (dx - np.transpose(dx, (1, 0, 2))) ** 2
        #         dy = (dy - np.transpose(dy, (1, 0, 2))) ** 2
        #         dz = (dz - np.transpose(dz, (1, 0, 2))) ** 2

        #         occ = np.ones(C.shape[1])
        #         occ_matrix = np.expand_dims(occ, axis=0) @ np.expand_dims(occ, axis=1)

        #         tau_p = np.sum((dx + dy + dz).T * occ_matrix, axis=(1,2)) 


        #     tau_p /= (2*density_g)

        #     return tau_p

        # def avg_local_orb_energy(self, density, orbitals, eigvals, grid='spherical'):
        #     """
        #     Please look at OuCarter or mRKS method
        #     Generates average local orbital energy. Described by Staroverov
        #     J. Chem. Phys. 146, 084103. [Equations 4 and/or 6]
        #     $$
        #     e_tilde = 1/n(r) * [ \sum_i \varepsilon_i * | \phi_i(r) |^2 ]
        #     $$
        #     """

        #     points = self.assert_grid(grid)

        #     phis      = evaluate_basis(self.basis, points)
        #     density_g = self.density(density=density, grid=grid)
        #     e_tilde   = contract('xp, xo, xo, x, xp-> p', phis, 
        #                                                 orbitals, orbitals, eigvals, 
        #                                                 phis) / density_g

        #     return e_tilde

        # def external_tilde(self, grid='spherical', method='grid'):
        #     """
        #     Please look at OuCarter or mRKS method
        #     Generates effective external potential from LDA exchange. Described by Ou + Carter. 
        #     J. Chem. Theory Comput. 2018, 14, 11, 5680–5689

        #     $$
        #     v^{~}{ext}(r) = \epsilon^{-LDA}(r) - \frac{\tau^{LDA}{L}}{n^{LDA}(r)}
        #     - v_{H}^{LDA}(r) - v_{xc}^{LDA}(r)
        #     $$
        #     (22) in [1].
        #     """

        #     points = self.assert_grid(grid)

        #     # LDA results
        #     Da0, Db0 = self.mf.make_rdm1()
        #     Ca0, Cb0 = self.mf.mo_coeff
        #     ea0, eb0 = self.mf.mo_energy

        #     da0_g = self.density(Da0, grid)
        #     db0_g = self.density(Db0, grid)

        #     # LDA exchange
        #     cx = -(3/np.pi)**(1/3)
        #     vxca = cx * da0_g ** (1/3)
        #     vxcb = cx * db0_g ** (1/3)

        #     # External tilde
        #     e_tilde = self.avg_local_orb_energy(Da0, Ca0, ea0, grid=grid)
        #     lap     = self.laplacian_density(Da0, grid=grid)
        #     grad    = self.gradient_density(Da0, grid=grid)
        #     grad    = grad[:,0] + grad[:,1] + grad[:,2]
        #     hartree = self.hartree(Da0, grid=grid)
            
        #     tau_l  = self.kinetic_energy_density_pauli(Ca0, grid=grid, method=method)
        #     tau_l += - 0.25 * lap + np.abs( grad )**2 / (8*da0_g) 
        
        #     external_tilde = e_tilde - tau_l/da0_g - hartree - vxca

        #     return external_tilde
        
        # Specialized for methods. 
        # def posdef_kinetic_energy_density(self, density, grid='spherical'):
        #     """
        #     Please look at OuCarter or mRKS method
        #     Evaluates the positive-definite kinetic energy density on grid
        #     t = 1/2 \nabla \cdot \nabla \gamma(r,r') 
        #     """

        #     points = self.assert_grid(grid)
        #     t = evaluate_posdef_kinetic_energy_density(density, self.basis, points)

        #     return t

        # def kinetic_energy_density(self, density, alpha=-1/4, grid='spherical'):
        #     """
        #     Please look at OuCarter or mRKS method
        #     Evaluates the general form of the kinetic energy density
        #     t = 1/2 \nabla \cdot \nabla \gamma(r,r') + alpha \nabla^2 n(r)
        #     """

        #     points = self.assert_grid(grid)
        #     t = evaluate_general_kinetic_energy_density(density, self.basis, points, alpha=alpha)

        #     return t

        # def kinetic_energy_density_pauli(self, C, grid='spherical', method='grid'):
        #     """
        #     Please look at OuCarter or mRKS method
        #     Obtains kinetic energy density in terms of the Pauli kinetic energy density
            
        #     Parameters
        #     ----------
        #     C: np.ndarray
        #         Occupied Molecular Orbitals
        #     """

        #     points = self.assert_grid(grid)

        #     density = C @ C.T
        #     density_g = self.density(density, grid=grid)

        #     basis_dx = self.ao_deriv(derivs=[1,0,0], transform=None, grid=grid)
        #     basis_dy = self.ao_deriv(derivs=[0,1,0], transform=None, grid=grid)
        #     basis_dz = self.ao_deriv(derivs=[0,0,1], transform=None, grid=grid)

        #     if method == 'grid':
        #         orbs = self.orbitals(C, grid=grid)
        #         d_orbs = ((basis_dx + basis_dy + basis_dz).T @ C).T
        #         tau_p = np.zeros_like( density_g )
        #         for i in range(C.shape[1]):
        #             for j in range(C.shape[1]):
        #                 if i == j:
        #                     pass
        #                 else:
        #                     tau_p += np.abs( orbs[i,:] * (d_orbs[j,:]) - orbs[j,:] * (d_orbs[i,:]) )**2

        #     elif method == 'basis':
        #         basis = self.ao_deriv(grid=grid)
        #         dx = contract('pm,mi,nj,pn->ijp', basis.T, C, C, basis_dx.T)
        #         dy = contract('pm,mi,nj,pn->ijp', basis.T, C, C, basis_dy.T)
        #         dz = contract('pm,mi,nj,pn->ijp', basis.T, C, C, basis_dz.T)
            
        #         dx = (dx - np.transpose(dx, (1, 0, 2))) ** 2
        #         dy = (dy - np.transpose(dy, (1, 0, 2))) ** 2
        #         dz = (dz - np.transpose(dz, (1, 0, 2))) ** 2

        #         occ = np.ones(C.shape[1])
        #         occ_matrix = np.expand_dims(occ, axis=0) @ np.expand_dims(occ, axis=1)

        #         tau_p = np.sum((dx + dy + dz).T * occ_matrix, axis=(1,2)) 


        #     tau_p /= (2*density_g)

        #     return tau_p

        # def avg_local_orb_energy(self, density, orbitals, eigvals, grid='spherical'):
        #     """
        #     Please look at OuCarter or mRKS method
        #     Generates average local orbital energy. Described by Staroverov
        #     J. Chem. Phys. 146, 084103. [Equations 4 and/or 6]
        #     $$
        #     e_tilde = 1/n(r) * [ \sum_i \varepsilon_i * | \phi_i(r) |^2 ]
        #     $$
        #     """

        #     points = self.assert_grid(grid)

        #     phis      = evaluate_basis(self.basis, points)
        #     density_g = self.density(density=density, grid=grid)
        #     e_tilde   = contract('xp, xo, xo, x, xp-> p', phis, 
        #                                                 orbitals, orbitals, eigvals, 
        #                                                 phis) / density_g

        #     return e_tilde

        # def external_tilde(self, grid='spherical', method='grid'):
        #     """
        #     Please look at OuCarter or mRKS method
        #     Generates effective external potential from LDA exchange. Described by Ou + Carter. 
        #     J. Chem. Theory Comput. 2018, 14, 11, 5680–5689

        #     $$
        #     v^{~}{ext}(r) = \epsilon^{-LDA}(r) - \frac{\tau^{LDA}{L}}{n^{LDA}(r)}
        #     - v_{H}^{LDA}(r) - v_{xc}^{LDA}(r)
        #     $$
        #     (22) in [1].
        #     """

        #     points = self.assert_grid(grid)

        #     # LDA results
        #     Da0, Db0 = self.mf.make_rdm1()
        #     Ca0, Cb0 = self.mf.mo_coeff
        #     ea0, eb0 = self.mf.mo_energy

        #     da0_g = self.density(Da0, grid)
        #     db0_g = self.density(Db0, grid)

        #     # LDA exchange
        #     cx = -(3/np.pi)**(1/3)
        #     vxca = cx * da0_g ** (1/3)
        #     vxcb = cx * db0_g ** (1/3)

        #     # External tilde
        #     e_tilde = self.avg_local_orb_energy(Da0, Ca0, ea0, grid=grid)
        #     lap     = self.laplacian_density(Da0, grid=grid)
        #     grad    = self.gradient_density(Da0, grid=grid)
        #     grad    = grad[:,0] + grad[:,1] + grad[:,2]
        #     hartree = self.hartree(Da0, grid=grid)
            
        #     tau_l  = self.kinetic_energy_density_pauli(Ca0, grid=grid, method=method)
        #     tau_l += - 0.25 * lap + np.abs( grad )**2 / (8*da0_g) 
        
        #     external_tilde = e_tilde - tau_l/da0_g - hartree - vxca

        #     return external_tilde

