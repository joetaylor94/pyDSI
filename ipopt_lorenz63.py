"""
Example file for the generation of an executable
for the determination of the dynamical state and 
parameters of a chaotic Lorenz system (1963).

Using this package, a set of coupled differential 
equations are transformed to discrete-time mappings 
and imposed as (strong) constraints on a cost function 
which defines a misfit between data and a model of 
the system.

"""

from methods import DynamicalSystem
from outputfiles import BuildIpoptCpp 

# Initialize dynamical system
Lorenz = DynamicalSystem()

# Define system via text file, indicating file path, state space dimension 
# and number of system parameters
Lorenz.define_system('dyn_sys.txt',3,3)

# Give indices of state variables to be observed and indices of 
# state vars to be coupled with control term.
meas_idx = [0,1,2]
ctrl_idx = [0]
Lorenz.set_observations(meas_idx,ctrl_idx, ('Xobs',"../data/xdata.txt",'uu'),('Yobs',"../data/ydata.txt"),('Zobs',"../data/zdata.txt"))

# Discretize continuous-time equations of motion with hermite polynomial
# interpolation as an additional constraint.
Lorenz.simpsonify(hermite=True)

# Define path of file containing variable / parameter bounds.
Lorenz.set_bounds(path='../data/bounds.txt')

#Lorenz.set_initial('../data/initdata.dat')

# Define name and path of forcing functions
#Lorenz.set_forcing(('Iinj','../data/Iinj.dat'))

# Calculate the constraint Jacobian and Lagrangian Hessian 
# to be provided to IPOPT.
Lorenz.get_ipopt_hessian()

# Build set of C++ files to be compiled.
Lorenz_build = BuildIpoptCpp(Lorenz, 'HH_Cond', 'problem_info.txt')
#Lorenz_build.build_make('/opt/Ipopt-source/build/lib')
