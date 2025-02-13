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
from gbasis.integrals.electron_repulsion import electron_repulsion_integral
from gbasis.contractions import GeneralizedContractionShell

import numpy as np
from opt_einsum import contract

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
        def __init__(self, mol, pbs_mol=None, basis_str=None, ref=None):
           self.mol = mol
        # Handle user-defined basis set
           self.basis = vlx.MolecularBasis.read(mol, basis_str)
           self.pbs = vlx.MolecularBasis.read(pbs_mol,basis_str) if pbs_mol else None
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
           except AttributeError:
            print("Error: 'get_nuclear_coordinates' not found in the Molecule class.")
           # Perform SCF Calculation based on ref
           if self.ref == 1:
             scf_drv = vlx.ScfRestrictedDriver()
           else:
             scf_drv = vlx.ScfUnrestrictedDriver()
           # Perform a quick LDA calculation to generate density matrices.
           self.scf_results = scf_drv.compute(mol, self.basis)
           # Extract density matrix from SCF results
           self.Da = self.scf_results['D_alpha']
           if self.ref != 1:
            self.Db = self.scf_results['D_beta']

           grid_drv = GridDriver()
           grid_drv.set_level(5)
           molgrid = grid_drv.generate(mol) # Generate grid for the molecule
           self.molgrid = molgrid
           print("hello spherical",dir(molgrid))
           # Step 2: Access grid points and weights
           x_coords = molgrid.x_to_numpy()  # Get x coordinates as a NumPy array
           y_coords = molgrid.y_to_numpy()  # Get y coordinates as a NumPy array
           z_coords = molgrid.z_to_numpy()  # Get z coordinates as a NumPy array
           coords = np.vstack((x_coords, y_coords, z_coords)).T  # Combine into a single array of coordinates
           self.spherical_points = coords  # Get grid points as NumPy array
           self.w = molgrid.w_to_numpy()  # Get weights for integration

        def assert_grid(self, grid):
            if grid == 'spherical':
                points = self.spherical_points
            elif grid == 'rectangular':
                assert self.rectangular_grid is not None, "Rectangular Grid must be defined first"
                points = self.rectangular_grid
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
                Total density on the requested grid (alpha + beta if provided).
            """
            n_elec = []
            # Ensure the grid is valid and fetch the grid points
            print("Da shape:", Da.shape)
            print("self_ao", self.to_ao())
            print("Type of self.to_ao:", type(self.to_ao()))

            #self.to_ao = np.array(self.to_ao())
            print("self.to_ao shape:", self.to_ao().shape)

            # determine the density on the grid points
            G = np.einsum("ab,bg->ag", Da, self.to_ao())
            n_g = np.einsum("ag,ag->g", self.to_ao(), G)
            #n_elec.append(np.dot(self.w, n_g))
            density_a =  n_g         # evaluate_density(Da, self.basis, points)
            
            # Initialize total density
            density_g = density_a
            
            # Add beta density if provided
            if Db is not None:
                G = np.einsum("ab,bg->ag", Db, self.to_ao())
                n_g = np.einsum("ag,ag->g", self.to_ao(), G)
                #n_elec.append(np.dot(self.w, n_g))
                density_b = n_g                     #evaluate_density(Db, self.basis, points)
                density_g += density_b  # Sum densities instead of concatenating
            
            return density_g

        def hartree(self, Da, Db=None, grid='spherical'):
            """
            Compute the Hartree integral of the density on the grid points with the interaction term 1/(r - r').
        
            Parameters:
            - Da: np.ndarray, density matrix for alpha electrons in AO basis
            - Db: np.ndarray, optional, density matrix for beta electrons (only used if provided)
        
            Returns:
            - result: float, the evaluated result of the Hartree integral.
            """
        
            # Step 2: Compute the density on the grid
            if Db is None:
                density_grid = self.density(Da)  # Compute density only for alpha
            else:
                density_grid = self.density(Da, Db)  # Compute total density
        
            # Step 3: Initialize the result array
            result = np.zeros(self.spherical_points.shape[0])  # One value per grid point
        
            # Step 4: Loop over grid points
            for i, r in enumerate(self.spherical_points):  # Loop over r
                for j, r_prime in enumerate(self.spherical_points):  # Loop over r'
                    if i == j:
                        continue  # Skip self-interaction (r == r')
        
                    # Compute distance |r - r'|
                    r_diff = r - r_prime
                    r_dist = np.linalg.norm(r_diff)
        
                    # Interaction term 1 / |r - r'|
                    interaction_term = 1 / r_dist
        
                    # Density at r'
                    density_at_r_prime = density_grid[j]  
        
                    # Compute contribution to the integral
                    result[i] += density_at_r_prime * interaction_term * self.w[j]  # Weighted sum
        
            # Step 5: Sum over all grid points
            total_result = -np.sum(result)  # Applying the final sum and sign convention
        
            return total_result
        '''
        def external(self, grid='spherical'):
            """
            Computes External Potential on grid.
        
            Parameters
            ----------
            grid: str
                Type of grid used. Default spherical.
                If 'rectangular' used, self.rectangular_grid != None
        
            Returns
            -------
            external_potential: np.ndarray
                External potential on the given grid.
            """
            # Generate the grid for the molecule
        
        
            # Compute external potential
            old_settings = np.seterr(divide="ignore")  # Silence warning for dividing by zero
            external_potential = np.sum(
                self.atomic_charges[None, :] / np.linalg.norm(self.spherical_points[:, :, None] - self.atomic_coords.T[None, :, :], axis=1),
                axis=1
            )
            np.seterr(**old_settings)
        
            return -external_potential
        '''
        def external(self, grid='spherical', threshold_dist=0.0):
            """
            Computes External Potential on grid, with the option to zero out potentials
            for elements that are too close to the nucleus based on a threshold distance.
        
            Parameters
            ----------
            grid: str
                Type of grid used. Default spherical.
                If 'rectangular' used, self.rectangular_grid != None
            threshold_dist: float or list, optional
                Threshold distance to zero out potentials for elements too close to the nucleus.
                If not provided, no zeroing occurs.
        
            Returns
            -------
            external_potential: np.ndarray
                External potential on the given grid.
            """
            # Generate the grid for the molecule
        
            # Compute external potential
            old_settings = np.seterr(divide="ignore")  # Silence warning for dividing by zero
            external_potential = (self.atomic_charges[None, :] / np.sum((np.linalg.norm(self.spherical_points[:, :, None] - self.atomic_coords.T[None, :, :])**2))**0.5)
            
            # Zero out potentials of elements that are too close to the nucleus
            external_potential[external_potential > 1.0 / np.array(threshold_dist)] = 0 
            # Restore old settings
            np.seterr(**old_settings)
            external_potential = -np.sum(external_potential, axis=1)

            # Sum over potentials for each dimension (if necessary)
            #external_potential = -np.sum(external_potential)
        
            return external_potential[0]


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

        def to_ao(self, grid='spherical'):
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
            xc_drv = vlx.XCIntegrator()
            chi_g = np.array(xc_drv.compute_gto_values(self.mol, self.basis, self.molgrid))
            '''
            points = self.assert_grid(grid)

            phis = evaluate_basis(self.basis, points)
            f_nm = contract( 'pb, p,p,pa->ab', phis.T, f_g, self.w, phis.T )
            f_nm = 0.5 * (f_nm + f_nm.T)
            '''
            return chi_g
        
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

