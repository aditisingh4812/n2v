"""
grider_veloxchem.py
Grider for VeloxChem
"""
import veloxchem 
from gbasis.evals.density import (
    evaluate_density,
    evaluate_density_laplacian,
    evaluate_density_gradient,
)
from gbasis.evals.eval import evaluate_basis
from gbasis.evals.eval_deriv import evaluate_deriv_basis
from gbasis.evals.electrostatic_potential import point_charge_integral


import numpy as np
from opt_einsum import contract

try:
    import veloxchem
    has_veloxchem = True
except ImportError:
    has_veloxchem = False

if has_veloxchem:
    import veloxchem as vlx
    import numpy as np
    from gbasis.evals.eval import evaluate_basis
    from gbasis.evals.electrostatic_potential import point_charge_integral

    class VeloxchemGrider:
        def __init__(self, mol, pbs_mol=None, basis_str=None, basis_file=None, ref=None):
           self.mol = mol
        # Handle user-defined basis set
           if basis_str:
             print(f"Basis being passed to MolecularBasis.read: {basis_str} (Type: {type(basis_str)})")
             if not isinstance(basis_str, str):
                raise ValueError(f"Expected basis_str to be a string, but got {type(basis_str)}")
             self.basis = vlx.MolecularBasis.read(mol, basis_str)
           elif basis_file:
               self.basis = vlx.MolecularBasis.read_from_file(basis_file, mol)
           else:
               self.basis = vlx.MolecularBasis.read(mol, "def2-SVP")  # Default to def2-SVP if no input provided
           self.pbs = vlx.MolecularBasis.read(pbs_mol, "def2-SVP") if pbs_mol else None
           self.ref = ref
           try:
            self.atomic_charges = self.mol.get_charge()  # Replace with the correct method if necessary
           except AttributeError:
            print("Error: 'get_nuclear_charges' not found in the Molecule class.")
           try:
            # Assuming that you want to get the coordinates for all atoms
            self.atomic_coords = [self.mol.get_atom_coordinates(i) for i in range(self.mol.number_of_atoms())]
           except AttributeError:
            print("Error: 'get_nuclear_coordinates' not found in the Molecule class.")
           #self.atomic_charges = self.mol.nuclear_charges()
           #self.atomic_coords = self.mol.nuclear_coordinates()
           # Perform SCF Calculation based on ref
           if self.ref == 1:
             scf_drv = vlx.ScfRestrictedDriver()
           else:
             scf_drv = vlx.ScfUnrestrictedDriver()
           # Perform a quick LDA calculation to generate density matrices.
           self.scf_results = scf_drv.compute(mol, self.basis)
           print(self.scf_results.keys())

           # Extract density matrix from SCF results
           self.Da = self.scf_results['D_alpha']
           if self.ref != 1:
            self.Db = self.scf_results['D_beta']
           # Generate a uniform rectangular grid manually
           x = np.linspace(0, 10, 10)  # Example grid with 10 points from 0 to 10
           y = np.linspace(0, 10, 10)
           z = np.linspace(0, 10, 10)
           self.rectangular_grid, self.w = self.generate_grid(x,y,z)
        def generate_grid(self, grid_spacing=0.2):
            """
            Generates a simple rectangular grid.
            """
            min_bounds = np.min(self.atomic_coords, axis=0) - 2.0
            max_bounds = np.max(self.atomic_coords, axis=0) + 2.0

            x = np.arange(min_bounds[0], max_bounds[0], grid_spacing)
            y = np.arange(min_bounds[1], max_bounds[1], grid_spacing)
            z = np.arange(min_bounds[2], max_bounds[2], grid_spacing)

            grid_points = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)
            weights = np.full(len(grid_points), grid_spacing**3)

            return grid_points, weights

        def assert_grid(self, grid_type):
            """
            Asserts the type of grid (spherical or rectangular) and returns the corresponding points.
            Parameters
            ----------
            grid_type : str
            The type of grid to use ('spherical' or 'rectangular').
            Returns
            -------
            np.ndarray
            The grid points corresponding to the requested grid type.
            Raises
            ------
            ValueError
            If the grid type is not recognized or if the rectangular grid is not defined.
            """
            if grid_type == 'spherical':
             # VeloxChem stores spherical grid points, so return them here.
             points = self.spherical_points
            elif grid_type == 'rectangular':
             if self.rectangular_grid is None:
              raise ValueError("Rectangular grid must be defined first. Please generate the grid before accessing it.")
             # Return the rectangular grid points.
             points = self.rectangular_grid
            else:
              raise ValueError("Invalid grid type specified. Use either 'spherical' or 'rectangular'.")
            return points
        def generate_grid(self, x, y, z):
            """
            Generates a cubic mesh grid from 3 separate linear spaces (x, y, z),
            and flattens the result into a 2D array.
            Parameters
            ----------
            x : np.ndarray
            The 1D array of x coordinates.
            y : np.ndarray
            The 1D array of y coordinates.
            z : np.ndarray
            The 1D array of z coordinates.
            Returns
            -------
            grid : np.ndarray
            A 2D array of shape (3, len(x)*len(y)*len(z)) containing the mesh points.
            shape : tuple
            A tuple containing the shape (len(x), len(y), len(z)) for the mesh grid.
            """
            # Generate a meshgrid from x, y, z using 'ij' indexing (which is typical for Cartesian grids)
            shape = (len(x), len(y), len(z))
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            # Flatten the meshgrid coordinates
            X = X.reshape((-1, 1))  # Reshape to (num_points, 1)
            Y = Y.reshape((-1, 1))  # Reshape to (num_points, 1)
            Z = Z.reshape((-1, 1))  # Reshape to (num_points, 1)
            # Concatenate the X, Y, Z coordinates to create a grid
            grid = np.concatenate((X, Y, Z), axis=1).T  # Shape: (3, num_points)
            return grid, shape
        def build_rectangular(self, npoints, overage=3.0):
            """
            Builds a rectangular grid that encompasses the molecule.
            Parameters
            ----------
            npoints: tuple
            Number of points per dimension (n_x, n_y, n_z)
            overage: float
            Spatial extent to extend the grid around the molecule (default: 3.0 Å)
            """
            # Get the atomic coordinates of the molecule
            atom_coords = self.mol.atom_coords()
            # Determine the minimum and maximum values for each dimension (x, y, z)
            xmin, xmax = np.min(atom_coords[:, 0]), np.max(atom_coords[:, 0])
            ymin, ymax = np.min(atom_coords[:, 1]), np.max(atom_coords[:, 1])
            zmin, zmax = np.min(atom_coords[:, 2]), np.max(atom_coords[:, 2])
            # Add the overage (padding) around the molecule's bounding box
            xmin -= overage
            xmax += overage
            ymin -= overage
            ymax += overage
            zmin -= overage
            zmax += overage
            # Generate equally spaced points in each dimension
            x_grid = np.linspace(xmin, xmax, npoints[0])
            y_grid = np.linspace(ymin, ymax, npoints[1])
            z_grid = np.linspace(zmin, zmax, npoints[2])
            # Create the 3D meshgrid
            gx, gy, gz = np.meshgrid(x_grid, y_grid, z_grid)
            # Flatten the meshgrid into a 2D array (each row is a 3D point)
            g3d = np.vstack([gx.ravel(), gy.ravel(), gz.ravel()]).T
            # Store the grid and the grid extents
            self.x = x_grid
            self.y = y_grid
            self.z = z_grid
            self.rectangular_grid = g3d
        def density(self, Da, Db=None, grid='spherical'):
            """
            Computes the density on the grid.
            Parameters
            ----------
            Da : np.ndarray
            Density matrix in AO basis for alpha electrons.
            Db : np.ndarray, optional
            Density matrix in AO basis for beta electrons (only used if provided).
            grid : str, optional
            Type of grid to use. Default is 'spherical'. If 'rectangular' is chosen,
            `self.rectangular_grid` must be defined.
            Returns
            -------
            density_g : np.ndarray
            Density on the requested grid.
            """
            # Ensure the grid is valid and fetch the grid points
            points = self.assert_grid(grid)
            # Evaluate the density using the provided density matrices and basis
            density_a = evaluate_density(Da, self.basis, points)
            # If beta density is provided, compute density for both alpha and beta
            if Db is not None:
             density_b = evaluate_density(Db, self.basis, points)
             density_g = np.concatenate([density_a, density_b])
             return density_g
            else:
            # If no beta density is provided, return only alpha density
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

            hartree_potential = point_charge_integral(self.basis, 
                                                    points, 
                                                    -np.ones(points.shape[0]), 
                                                    transform=None, 
                                                    coord_type='spherical')

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
                basis = self.basis
            else:
                basis = self.pbs

            phis = evaluate_basis(basis, points)
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

            phis = evaluate_basis(self.basis, points)
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

            phis = evaluate_basis(self.basis, points)
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

            lap_density = evaluate_density_laplacian( density, self.basis, points )
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

            grad_density = evaluate_density_gradient(density, self.basis, points)
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

            orbs_deriv = evaluate_deriv_basis( self.basis, points, np.array(derivs), 
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

