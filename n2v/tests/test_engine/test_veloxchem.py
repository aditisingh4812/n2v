import n2v
import veloxchem as vlx
import pytest
import numpy as np

@pytest.fixture
def ine():
    # Create the VeloxChem molecule object
    molecule_data = """
    Ne
    """
    molecule = vlx.Molecule.read_xyz_string(molecule_data)

    # Create a molecular basis for the molecule
    basis = vlx.MolecularBasis.read('aug-cc-pvtz')  # Example basis set
    
    # Initialize the inverter object
    ine = n2v.Inverter(engine='veloxchem')

    # Set up the system in the inverter
    ine.set_system(molecule, basis)

    # Mock some example data (replace with actual calculation results)
    # In a real case, you would calculate or load these from the VeloxChem results.
    ine.Dt = [np.random.rand(5, 5), np.random.rand(5, 5)]  # Random data for testing
    ine.ct = [np.random.rand(5, 5), np.random.rand(5, 5)]  # Random data for testing
    ine.et = [np.random.rand(5), np.random.rand(5)]  # Random energy data

    return ine

def test_zmp(ine):
    ine.invert(method='zmp', guide_components='fermi_amaldi', lambda_list=[750], opt_max_iter=100, opt_tol=1e-5)
    assert np.isclose(ine.eigvecs_a[:5].all(),
                      np.array([-30.70166724,  -1.58399267,  -0.73369545,  -0.73369545, -0.73369545]).all())

def test_wuyang(ine):
    ine.invert("WuYang", guide_components="fermi_amaldi")
    assert np.isclose(ine.eigvecs_a[:5].all(),
                      np.array([-30.75604341,  -1.60572171,  -0.75614577,  -0.75614564, -0.75614546]).all())

def test_pedeco(ine):
    ine.invert("PDECO", opt_max_iter=200, guide_components="fermi_amaldi", gtol=1e-6)
    assert np.isclose(ine.eigvecs_a[:5].all(),
                      np.array([-30.75843463,  -1.60691725,  -0.75708326,  -0.75708173, -0.75708055]).all())

def test_oucarter_invalid_grid(ine):
    # Test case where the grid generation should raise an error
    x = np.linspace(-5, 10, 1501)
    y = [0]
    z = [0]
    grid, shape = ine.eng.grid.generate_grid(x, y, z)

    with pytest.raises(ValueError):
        ine.invert('OC', vxc_grid=grid)

def test_mrks_invalid_grid(ine):
    # Test case where the grid generation for mRKS method should raise an error
    x = np.linspace(-5, 10, 1501)
    y = [0]
    z = [0]
    grid, shape = ine.eng.grid.generate_grid(x, y, z)

    with pytest.raises(ValueError):
        ine.invert('mRKS', vxc_grid=grid, opt_max_iter=30, frac_old=0.8, init='scan')

