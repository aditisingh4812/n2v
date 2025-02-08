import psi4
from psi4.driver.procrouting.response.scf_response import tdscf_excitations
import numpy as np
import n2v
#import sys
#sys.path.append('/home/aditisingh4812/tc_files/calc/ks_pies/tddft/psi4/n2v1')  # Replace with your actual path

# Import inverter from n2v1 (adjust path based on your structure)
#from n2v1.inverter import Inverter

import matplotlib.pyplot as plt
# Set memory and threading
psi4.set_memory(int(4e9))  # 4 GB memory
psi4.set_num_threads(4)  # 4 threads
psi4.core.clean()

# Set output file
psi4.core.set_output_file("out.dat", True)
f=open('file.txt','w')
# Define the nitrogen molecule and custom basis set directly within the geometry block
Ne = psi4.geometry("""
Be 0 0 0
noreorient
nocom
symmetry c1
""" )
BASIS = 'cc-pvtz-decon'
# Set options for the SCF calculation
psi4.set_options({
    "reference": "rhf",       # Restricted Hartree-Fock reference
    "opdm": True,             # One-particle density matrix (OPDM)
    "SCF_TYPE": "DF",         # Density Fitting
    "basis" : BASIS,
#    "CC_TYPE": "DF",          # Coupled Cluster with Density Fitting
#    "tpdm": True,             # Two-particle density matrix (TPDM)
#    "cachelevel": 0,          # Minimal caching
#    "OPDM_RELAX": True,       # Relaxed OPDM
    'DFT_SPHERICAL_POINTS': 590,  # DFT spherical grid points
    'DFT_RADIAL_POINTS': 99,      # DFT radial grid points
    'save_jk': True           # Save J/K matrices (Coulomb and exchange matrices)
})

#wfn1 = psi4.properties("hf/cc-pvdz", return_wfn=True, molecule=Ne, properties=["dipole"])[1]
wfn = psi4.energy("hf", return_wfn=True, molecule=Ne)[1]
#######################initialize inverter manually###############

ca = np.asarray(wfn.Ca())
print("ca",ca)
da = np.asarray(wfn.Da())
print("da",da)
ea = np.asarray(wfn.epsilon_a())
print("ea",ea)


inv = n2v.Inverter.from_wfn(wfn, pbs=BASIS)
print("here is the old one")
pyscf_inv = n2v.Inverter(engine='psi4')
print("success")
epsilon = 5e-7
delta_rho = epsilon * np.eye(da.shape[0])
drho =  np.einsum('ij,gi,gj->g', delta_rho, ca, ca)
print("drho", drho)
# Define the grid
coords = np.array([(0., 0., x) for x in np.linspace(-5, 5, 43)])  # Custom grid points
exit()

da1 = da + delta_rho
# Overwrite the Density matrix
new_da_matrix = psi4.core.Matrix.from_array(da1)
np.save("dm.npy",new_da_matrix)
# Overwrite the Coefficients matrix
new_ca_matrix = psi4.core.Matrix.from_array(ca)
np.save("coeff.npy",new_ca_matrix)
# Overwrite the Eigenvalues (Orbital Energies)
new_ea_vector = psi4.core.Vector.from_array(ea)
np.save("eigen.npy",new_ea_vector)
# Overwrite the existing wavefunction with the new values
wfn.Da().copy(new_da_matrix)
wfn.Ca().copy(new_ca_matrix)
wfn.epsilon_a().copy(new_ea_vector)





#inv.invert("mrks",init=None,opt_max_iter=200 )
inv.invert("WuYang", opt_max_iter=100, opt_method="trust-exact", reg = 0 ,  gtol=1e-6,
           guide_components="fermi_amaldi")



# Build Grid

x = np.linspace(-5,5,43)
y = np.zeros_like(x)
z = np.zeros_like(x)
grid = np.array([x,y,z])

# Additionaly, one can use the generate grid function. 
grid2 = inv.eng.grid.generate_grid(x=x, y=[0], z=[0])[0]

# Get Hartree and Fermi-Amadli Potentials:

vH1 = inv.eng.grid.esp(Da=inv.Dt[0], Db=inv.Dt[1], grid=grid)[1]
vFA1 = (1-1/(inv.nalpha + inv.nbeta)) * vH1

vrest1 = inv.eng.grid.ao(inv.v_pbs, grid=grid, basis=inv.eng.pbs)  # Note that specify the basis set 
                                                                  # that vrest is on.
    
# Compute vxc according to the previous equation. 
vxc1 = vFA1 + vrest1 - vH1
########################################################################################################
exit()

inv1 = Inverter.from_wfn(wfn, pbs=BASIS)


da2 = da - delta_rho
# Overwrite the Density matrix
new_da_matrix1 = psi4.core.Matrix.from_array(da2)
np.save("dm1.npy",new_da_matrix1)
# Overwrite the Coefficients matrix
new_ca_matrix = psi4.core.Matrix.from_array(ca)
np.save("coeff1.npy",new_ca_matrix)
# Overwrite the Eigenvalues (Orbital Energies)
new_ea_vector = psi4.core.Vector.from_array(ea)
np.save("eigen1.npy",new_ea_vector)
# Overwrite the existing wavefunction with the new values
wfn.Da().copy(new_da_matrix)




inv1.invert("mrks",init=None,opt_max_iter=200 )
#inv1.invert("WuYang", opt_max_iter=100, opt_method="SLSQP", reg = 0,  gtol=1e-6)
         #  guide_components="LDA")



# Build Grid

x = np.linspace(-5,5,43)
y = np.zeros_like(x)
z = np.zeros_like(x)
grid = np.array([x,y,z])

# Additionaly, one can use the generate grid function.
grid2 = inv1.eng.grid.generate_grid(x=x, y=[0], z=[0])[0]

# Get Hartree and Fermi-Amadli Potentials:

vH2 = inv1.eng.grid.esp(Da=inv1.Dt[0], Db=inv1.Dt[1], grid=grid)[1]
vFA2 = (1-1/(inv1.nalpha + inv1.nbeta)) * vH2

vrest2 = inv1.eng.grid.ao(inv1.v_pbs, grid=grid, basis=inv1.eng.pbs)  # Note that specify the basis set 
                                                                  # that vrest is on.
    
# Compute vxc according to the previous equation. 
vxc2 = vFA2 + vrest2 - vH2
vxc_diff = vxc1 - vxc2 

###### For loop to write the coordinates and Density###
s=0
for x in np.linspace(-5,5,43):
 y = round(x,6)
 f.write(str(y)+"\t\t\t"+str(vxc_diff[s])+"\t\t\t"+ str(drho[s]) +"\t\t\t" + str((vxc_diff[s])/(2*drho[s])) +"\n")
 s+=1
f.close()
