import numpy as np
import n2v
import veloxchem as vlx
import matplotlib.pyplot as plt

molecule_data = """2

H    0.000000000000        0.740848095288        0.582094932012
H    0.000000000000       -0.740848095288        0.582094932012
"""

# Read molecule data into VeloxChem's Molecule object
molecule = vlx.Molecule.read_xyz_string(molecule_data)

# Now you can set the system
basis = 'sto-3g'
ref = 1

# Compute SCF results using VeloxChem
scf_drv = vlx.ScfRestrictedDriver()
scf_drv.ostream.mute()  # Mute output for cleaner logs
scf_results = scf_drv.compute(molecule, basis)  # This returns the results as a dictionary

exit()
# Initialize the inverter for VeloxChem
inv = n2v.Inverter(engine='veloxchem')
# Now, pass scf_results to set_system
inv.set_system(molecule, basis, ref=ref, pbs='same', scf_results=scf_results)

# Now you can proceed with the inversion or other methods
inv.invert("WuYang", opt_max_iter=100, opt_method="trust-exact", reg=0, gtol=1e-6, guide_components="fermi_amaldi")

