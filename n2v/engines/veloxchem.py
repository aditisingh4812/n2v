"""
Provides interface n2v interface to Veloxchem
"""


from .engine import Engine
import numpy as np
from opt_einsum import contract

try:
    import veloxchem 
    has_veloxchem = True
except ImportError:
    has_veloxchem = False
    
if has_veloxchem:
    from ..grid import VeloxchemGrider
    from veloxchem import GridDriver, XCIntegrator
class VeloxchemEngine(Engine):
       def __init__(self, engine='veloxchem', ref=1):
        self.engine = engine
        self.ref = ref  # Initialize the reference type (1 for restricted, 2 for unrestricted)
        # Other initializations
        """
        Veloxchem Engine Class
        """
       def set_system(self, xyz_string, basis, ref=1, pbs='same',scf_results=None):
        print(f"Basis being used: {basis}, Type: {type(basis)}")
        """
        Initializes geometry and basis information.
        Parameters
        ----------
        xyz_string: str
            XYZ format string defining the molecular geometry.
        basis: str
            Basis set for calculation.
        ref: int
            Reference: Restricted (1) or Unrestricted (2).
        pbs: str
            Basis set for potential used (default: same as `basis`).
        scf_results: dict, optional (default=None)
        SCF results returned from VeloxChem (from `scf_drv.compute()`) to avoid recomputation.
        """
        # Define the molecule
        self.mol = veloxchem.Molecule.read_xyz_string(xyz_string)
        print(f"Basis type before MolecularBasis.read: {type(basis)}")
        # Assign basis sets
        self.basis = veloxchem.MolecularBasis.read(self.mol, basis)

        self.pbs = veloxchem.MolecularBasis.read(self.mol, basis if pbs == 'same' else pbs)
        # Store reference type
        self.basis_str = basis

        self.pbs_str = basis if pbs == 'same' else pbs
        # Print the basis and pbs information for debugging
        print(f"Basis string: {self.basis_str}, Type: {type(self.basis_str)}")
        print(f"PBS string: {self.pbs_str}, Type: {type(self.pbs_str)}")
        #print(f"Basis being passed to MolecularBasis.read: {basis_str} (Type: {type(basis_str)})")

        # Get number of alpha and beta electrons
        self.nalpha = self.mol.number_of_alpha_electrons()
         # Get number of alpha and beta electrons
        self.nbeta = self.mol.number_of_beta_electrons()
        # Perform SCF Calculation
        if self.ref == 1:
            self.scf_drv = veloxchem.ScfRestrictedDriver()
        else:
            self.scf_drv = veloxchem.ScfUnrestrictedDriver()  # Use unrestricted driver if ref=2
        if scf_results is None:
          print("Performing SCF calculation...")
          self.scf_results = self.scf_drv.compute(self.mol, self.basis)
        else:
          print("Using provided SCF results.")
          self.scf_results = scf_results
        print("this is done")
       def initialize(self):
        """
        Initializes basic objects required for the VeloxchemEngine.
        """
        # Initialize the basis sets using the VeloxChem interface.
        self.basis = veloxchem.MolecularBasis.read(self.mol, self.basis_str, ostream=None)
        print(f"PBS string: {self.basis_str}, Type: {type(self.basis)}")
        self.pbs   = veloxchem.MolecularBasis.read(self.mol, self.pbs_str, ostream=None)
        # Retrieve dimensions for each basis set.
        print(f"PBS string: {self.pbs_str}, Type: {type(self.pbs)}")
        self.nbf   = self.basis.get_dimension_of_basis(self.mol)
        self.npbs  = self.pbs.get_dimension_of_basis(self.mol)
        # Initialize the grid using the VeloxchemGrider.
        #self.grid = VeloxchemGrider(self.mol, self.basis, self.ref)
        self.grid = VeloxchemGrider(self.mol, basis_str=self.basis_str, ref= self.ref)

       def get_T(self):
        """Kinetic Potential in ao basis"""
        return veloxchem.KineticEnergyIntegralsDriver().compute(self.mol, self.basis).to_numpy()
       def get_Tpbas(self):
        """Kinetic Potential in pbs"""
        return veloxchem.KineticEnergyIntegralsDriver().compute(self.mol, self.pbs).to_numpy()
       def get_V(self):
        """External potential in ao basis"""
        return veloxchem.NuclearPotentialIntegralsDriver().compute(self.mol,self.basis).to_numpy()
       def get_A(self):
        """Inverse squared root of S matrix computed via eigenvalue decomposition."""
        # Compute the overlap matrix S in the atomic orbital basis.
        S = veloxchem.OverlapIntegralsDriver().compute(self.mol, self.basis).to_numpy()
        # Diagonalize S.
        eigvals, eigvecs = np.linalg.eigh(S)
        # Define a threshold to avoid division by zero.
        threshold = 1e-16
        eigvals_inv_sqrt = np.array([1/np.sqrt(val) if val > threshold else 0 for val in eigvals])
        # Reconstruct the inverse square root matrix.
        A = eigvecs @ np.diag(eigvals_inv_sqrt) @ eigvecs.T
        print("this is done")
        return A
       def get_S(self):
        print("problem starts")

        """Overlap matrix in AO basis"""
        return veloxchem.OverlapIntegralsDriver().compute(self.mol,self.basis).to_numpy()
       def get_S3(self, grid_level=4):
        """
        Builds the 3-index overlap matrix.
        Manually built since VeloxChem (like PySCF) may not support it directly.
        Returns
        -------
        S3 : np.ndarray
        Shape: (nbf, nbf, nbf) if using the same basis for all integrals,
                or (nbf, nbf, npbs) if a separate potential basis is used.
        """

        # Step 1: Generate the grid using VeloxChem
        grid_drv = GridDriver()
        grid_drv.set_level(grid_level)  # Adjust the grid density level (1-8, default is 4)
        molgrid = grid_drv.generate(self.mol)  # Generate molecular grid
        x_coords = molgrid.x_to_numpy()  # Get x coordinates as a NumPy array
        y_coords = molgrid.y_to_numpy()  # Get y coordinates as a NumPy array
        z_coords = molgrid.z_to_numpy()  # Get z coordinates as a NumPy array
        coords = np.vstack((x_coords, y_coords, z_coords)).T  # Combine into a single array of coordinates
        weights = molgrid.w_to_numpy()  # Extract integration weights
        # Step 2: Compute AO basis functions on the grid
        xc_drv = XCIntegrator()
        bs1 = xc_drv.compute_gto_values(self.mol, self.basis, molgrid)  # Correct AO evaluation
        bs2 = xc_drv.compute_gto_values(self.mol, self.pbs, molgrid)
        print(f"Shape of bs1: {bs1.shape}")
        print(f"Shape of bs2: {bs2.shape}")
        print(f"Shape of weights: {weights.shape}")
        # Ensure shapes are correct for contraction
        bs1 = bs1.T  # Reshape so that bs1 is (num_points, nbf), num_points = 2
        bs2 = bs2.T  # Reshape bs2 similarly if it's not already
        weights = weights.reshape(-1)  # Ensure weights is a 1D array (size should be the same as num_points)
        # Step 3: Compute 3-index overlap matrix S3 (if pbs_str is 'same')
        if self.pbs_str == 'same':
          S3 = contract('ij, ik, il, i -> jkl', bs1, bs1, bs1, weights)
        else:
          # If a different potential basis is used, evaluate that as well.
          S3 = contract('ij, ik, il, i -> jkl', bs1, bs1, bs2, weights)
        return S3
       def get_S4(self):
        """
        Obtains a 4-index AO Overlap Matrix using Density Fitting.
        The procedure is:
        1. Evaluate the AO basis functions on a numerical grid.
        2. Create an auxiliary (density-fitting) basis by appending '-jk-fit' 
             to the primary basis identifier.
        3. Evaluate the auxiliary basis functions on the same grid.
        4. Form an intermediate three-index tensor by contracting the auxiliary 
             and primary basis evaluations.
        5. Compute the auxiliary basis overlap matrix and its pseudo-inverse.
        6. Contract these objects to yield the 4-index overlap matrix.
    
        Returns
        -------
        S4 : np.ndarray
         4-index overlap matrix. Its shape is (nbf, nbf, nbf, nbf) or
         (  nbf, nbf, nbf, npbs) if a separate potential basis is used.
        """
        #   - self.grid.weights: a NumPy array of integration weights.
        grid = self.grid
        # Evaluate the AO basis functions for the primary basis on the grid.
        bs1 = self.basis.eval_ao(grid.coords)  # shape: (n_points, nbf)
        auxbasis = self.basis_str + '-jk-fit'
        aux = veloxchem.MolecularBasis.read(self.mol, auxbasis, ostream=None)
        # Evaluate the auxiliary AO functions on the grid.
        bs2 = aux.eval_ao(grid.coords)  # shape: (n_points, n_aux)
        S_Pmn = contract('ij, ik, il, i -> jkl', bs2, bs1, bs1, grid.weights)
        S_PQ = aux.get_overlap()  # Expected to return a NumPy array.
        # Compute the pseudo-inverse of the auxiliary overlap matrix.
        S_PQinv = np.linalg.pinv(S_PQ, rcond=1e-9)
        # Contract the intermediates to form the 4-index overlap matrix.
        S4 = contract('Pmn,PQ,Qrs->mnrs', S_Pmn, S_PQinv, S_Pmn)
        return S4
       def generate_jk(self, gen_K=False):
        """
        Creates a JK-like object for the generation of Coulomb and Exchange matrices.
        In Psi4, this is done via psi4.core.JK.build. VeloxChem does not yet provide an
        equivalent interface, so we define a dummy class that you can later modify to
        perform the actual Coulomb and Exchange matrix evaluations.
        Parameters
        ----------
        gen_K : bool, optional
        Whether to generate the Exchange (K) matrix as well (default is False).
        Returns
        -------
        jk : object
        An object with methods to compute the Coulomb (and optionally, Exchange) matrices.
        """
        # Define a dummy JK-like class for VeloxChem.
        class VeloxchemJK:
            def __init__(self, basis):
             self.basis = basis
             self.do_K = False
             self.memory = 0  # memory in bytes, for example
            def memory_estimate(self):
             # Provide a crude memory estimate based on the basis size.
             # (This is a placeholder calculation; adjust as needed.)
             nbf = self.basis.get_dimension_of_basis()
             return nbf**2 * 8  # for instance, 8 bytes per float element
            def set_memory(self, mem):
             self.memory = mem
            def set_do_K(self, do_K):
             self.do_K = do_K
            def  initialize(self):
              # Perform any necessary initializations.
              # For example, precompute integral intermediates if available.
              pass
            def compute(self, density):
              """
              Compute the Coulomb (J) matrix, and if requested, the Exchange (K) matrix.
              Parameters
              ----------
              density : np.ndarray
                 The density matrix.
              Returns
              -------
              J : np.ndarray
                Coulomb matrix.
              K : np.ndarray (if self.do_K is True)
                Exchange matrix.
              Note: This method is a placeholder. You need to implement the actual
                  algorithms for evaluating the Coulomb and Exchange matrices.
              """
              # Placeholder implementation:
              # Here you should implement the integral evaluations and contractions
              # needed to obtain the Coulomb and Exchange matrices.
              raise NotImplementedError("JK computation has not been implemented for VeloxChem.")
              # Instantiate and set up the JK object.
        jk = VeloxchemJK(self.basis)
        memory = int(jk.memory_estimate() * 1.1)
        jk.set_memory(memory)
        jk.set_do_K(gen_K)
        jk.initialize()
        return jk
       def _compute_coulomb_exchange(self, Cocc_a, Cocc_b, eri_tensor):
        """
        Compute the Coulomb and Exchange matrices from the ERI tensor and occupied orbital coefficients.
        
        Parameters
        ----------
        Cocc_a : array-like
            Occupied orbital coefficients for alpha electrons.
        Cocc_b : array-like
            Occupied orbital coefficients for beta electrons.
        eri_tensor : np.ndarray
            The computed Electron Repulsion Integral (ERI) tensor.
        
        Returns
        -------
        J_alpha, J_beta : np.ndarray
            The Coulomb matrices for alpha and beta spins.
        """
        # Number of basis functions
        nbf = eri_tensor.shape[0]
        
        # Initialize the Coulomb (J) and Exchange (K) matrices
        J_alpha = np.zeros((nbf, nbf))
        J_beta = np.zeros((nbf, nbf))
    
        # Debugging the shapes of the arrays
        print(f"Shape of Cocc_a: {Cocc_a.shape}")
        print(f"Shape of Cocc_b: {Cocc_b.shape}")
        print(f"Shape of eri_tensor: {eri_tensor.shape}")
    
        # Loop over the indices to compute the Coulomb and Exchange integrals
        for i in range(nbf):
            for j in range(nbf):
                # Compute Coulomb term for alpha electrons
                for k in range(nbf):
                    for l in range(nbf):
                        J_alpha[i, j] += eri_tensor[i, j, k, l] * Cocc_a[k, 0] * Cocc_a[l, 0]
    
                # Compute Coulomb term for beta electrons
                for k in range(nbf):
                    for l in range(nbf):
                        J_beta[i, j] += eri_tensor[i, j, k, l] * Cocc_b[k, 0] * Cocc_b[l, 0]
    
        # Return the computed Coulomb matrices
        return J_alpha, J_beta
       '''
       def compute_hartree(self, Cocc_a=None, Cocc_b=None):
        """
        Generates Coulomb and Exchange matrices from occupied orbitals.
        If the occupied orbital coefficients (Cocc_a and Cocc_b) are not provided,
        they are computed using the ScfRestrictedDriver.
        Parameters
        ----------
        Cocc_a : array-like, optional
        Occupied orbital coefficients for alpha electrons.
        Cocc_b : array-like, optional
        Occupied orbital coefficients for beta electrons.
        Returns
        -------
        J : tuple of np.ndarray
        A tuple containing (J_alpha, J_beta), the Coulomb matrices for alpha and beta spins.
        """
        # Compute the occupied orbital coefficients if not provided.
        if Cocc_a is None or Cocc_b is None:
             scf_results = veloxchem.ScfRestrictedDriver().compute(self.mol, self.basis)
             Cocc_a = scf_results["C_alpha"]
             Cocc_b = scf_results["C_beta"]
        if not hasattr(self, 'jk') or self.jk is None:
           self.jk = CoulombExchangeOperator()  # Example initialization; replace with actual class
        # Add the occupied orbital contributions to the JK object.
        self.jk.C_left_add(Cocc_a)
        self.jk.C_left_add(Cocc_b)
        # Compute the Coulomb (and possibly Exchange) matrices.
        self.jk.compute()
        # Clear the accumulated contributions so that future computations are not affected.
        self.jk.C_clear()
        # Retrieve the computed matrices.
        # Assuming self.jk.J() returns a sequence (e.g., [J_alpha, J_beta]),
        # we convert these to NumPy arrays.
        J_results = self.jk.J()
        J = (np.array(J_results[0]), np.array(J_results[1]))
        return J
        '''
       def hartree_NO(self):
        """
        Computes the Hartree potential in the AO basis from Natural Orbitals.

        Returns
        -------
        J0 : tuple of np.ndarray
            The Coulomb (and possibly exchange) matrices computed from the natural orbital basis.

        Raises
        ------
        ValueError
            If SCF results are not available.
        """
        if not hasattr(self, "scf_results"):
            raise ValueError("SCF calculation has not been performed. Run set_system() first.")

        # Extract the alpha density matrix
        Dta = self.scf_results.Da()

        # Diagonalize the alpha density matrix
        eigvals, C_NO = np.linalg.eigh(Dta)

        # Sort eigenvalues and eigenvectors in descending order
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        C_NO = C_NO[:, order]

        # Compute square root of occupation numbers
        occ = np.sqrt(np.maximum(eigvals, 0))  # Ensure non-negative values

        # Construct natural orbital coefficient matrix
        new_CA = C_NO * occ

        # Validate reconstructed density
        if not np.allclose(new_CA @ new_CA.T, Dta, atol=1e-8):
            raise ValueError("Reconstructed density matrix does not match the input density matrix.")

        # Handle unrestricted case (beta density matrix)
        if self.ref == 1:
            new_CB = new_CA.copy()
        else:
            Dtb = self.scf_results.Db()
            eigvals_b, C_NO_b = np.linalg.eigh(Dtb)
            order_b = np.argsort(eigvals_b)[::-1]
            eigvals_b = eigvals_b[order_b]
            C_NO_b = C_NO_b[:, order_b]
            occ_b = np.sqrt(np.maximum(eigvals_b, 0))  # Ensure non-negative values
            new_CB = C_NO_b * occ_b

        # Compute Hartree potential (Coulomb matrix)
        J0 = self.compute_hartree(new_CA, new_CB)
        return J0
       def hartree_NO(self, Dta):
        """
        Computes the Hartree potential in the AO basis from Natural Orbitals.
        This method diagonalizes the (alpha) density matrix to obtain the natural
        orbitals and their occupations, reconstructs a square-root form of the density,
        and then computes the Hartree potential (e.g. Coulomb matrix) using these orbitals.
        Parameters
        ----------
        Dta : np.ndarray
        The α density matrix (in the AO basis) to be used for diagonalization.
        Returns
        -------
        J0 : tuple of np.ndarray
        The Coulomb (and possibly exchange) matrices computed from the natural orbital basis.
        Raises
        ------
        ValueError
        If no wavefunction (wfn) object is provided.
        """
        if self.wfn is None:
          raise ValueError("Please provide a wavefunction object to the Inverter, e.g., Inverter.eng = wfn")
        # For VeloxChem, assume that the density matrix is available as Dta (a NumPy array)
        # You might also have a method such as self.wfn.Da() to obtain it.
        # Diagonalize Dta using numpy.linalg.eigh (which returns eigenvalues in ascending order)
        eigvals, C_NO = np.linalg.eigh(Dta)
        # Sort the eigenvalues and eigenvectors in descending order so that the largest
        # eigenvalues (most occupied orbitals) come first.
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        C_NO = C_NO[:, order]
        # Compute the square root of the eigenvalues.
        # (This gives you the occupation amplitudes.)
        occ = np.sqrt(eigvals)
        # Reconstruct the “natural orbital coefficient” matrix by scaling each column of C_NO
        # by the corresponding square-root occupation number.
        # (Broadcasting multiplies each column by the corresponding occ value.)
        new_CA = C_NO * occ
        # Verify that the reconstructed density (new_CA @ new_CA.T) matches the input density Dta.
        if not np.allclose(new_CA @ new_CA.T, Dta, atol=1e-8):
             raise ValueError("Reconstructed density matrix does not match the input density matrix.")
        # For a restricted reference (self.ref == 1), alpha and beta orbitals are identical.
        if self.ref == 1:
             new_CB = new_CA.copy()
        else:
             # For an unrestricted case, assume that the beta density matrix is obtainable via self.wfn.Db()
             # (and that it returns a NumPy array). Then diagonalize similarly.
             Dtb = self.wfn.Db()
             eigvals_b, C_NO_b = np.linalg.eigh(Dtb)
             order_b = np.argsort(eigvals_b)[::-1]
             eigvals_b = eigvals_b[order_b]
             C_NO_b = C_NO_b[:, order_b]
             occ_b = np.sqrt(eigvals_b)
             new_CB = C_NO_b * occ_b
        # Now compute the Hartree potential (for example, the Coulomb matrices) using the
        # natural orbital coefficient matrices new_CA and new_CB.
        J0 = self.compute_hartree(new_CA, new_CB)
        return J0
       def run_single_point(self, mol, basis, method="HF"):
        """
        Run a single-point energy calculation in VeloxChem.
        Parameters
        ----------
        mol : vlx.Molecule
        The molecule object.
        basis : str
        The basis set to be used.
        method : str, optional
        The electronic structure method (default is "HF").
        Returns
        -------
        D : np.ndarray or tuple of np.ndarray
              Density matrix (Da for restricted, Da & Db for unrestricted).
        C : np.ndarray or tuple of np.ndarray
              Molecular orbital coefficient matrix (Ca for restricted, Ca & Cb for unrestricted).
        e : np.ndarray or tuple of np.ndarray
              Orbital energy levels (epsilon_a for restricted, epsilon_a & epsilon_b for unrestricted).
        """
        # Initialize SCF driver
        scf_drv = veloxchem.ScfRestrictedDriver() if self.ref == 1 else veloxchem.ScfUnrestrictedDriver()
        # Set up calculation
        scf_drv.xc_functional = method if "DFT" in method.upper() else None  # Assign DFT functional if applicable
        results = scf_drv.compute(mol, basis)
        # Extract Density, Coefficients, and Orbital Energies
        if self.ref == 1:
              D = results["D_alpha"]
              C = results["C_alpha"]
              e = results["e_alpha"]
        else:
              D = (results["D_alpha"], results["D_beta"])
              C = (results["C_alpha"], results["C_beta"])
              e = (results["e_alpha"], results["e_beta"])
        return D, C, e
 
