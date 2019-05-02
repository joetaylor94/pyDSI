# pyDSI

This python module uses statistical variational inference 
for predicting the evolution of physical dynamical systems.
The package generates C++ files required for the
solution of state and parameter estimation problems in the
open-source large-scale ​nonlinear optimization software
IPOPT. Equations of motion of user-defined dynamical systems 
are temporally discretized, and the Jacobian and Hessian 
matrices of the discrete system are calculated with symbolic 
differentiation. The equations of motion are imposed as 
strong constraints on the minimization of a cost function
defining the distance between the model and time series 
observations of the system to be estimated.
This code requires the installation of the following:
    
    - Sympy (.py)
    - Numpy (.py)
    - IPOPT 
    
IPOPT can be downloaded from https://projects.coin-or.org/Ipopt
This software package was used in the following publications
(in review):
    
    [1] J. Taylor et al., Stochasticity and convergence in data 
        assimilation of predictive neuron models
        
    [2] K.Abu-Hassan et al., Construction of neuromorphic models 
        of respiratory neurons
